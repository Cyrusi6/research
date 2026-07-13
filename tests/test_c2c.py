import json
import shutil
import subprocess
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import yaml

import auto_research.config as config_module
import auto_research.cli as cli_module
import auto_research.agents.literature as literature_module
import auto_research.agents.experiment as experiment_module
import auto_research.orchestrator as orchestrator_module
from auto_research.adapters.runner import ExperimentRunner
from auto_research.agents.debate import MultiAgentReasoningService
from auto_research.agents.experiment import ExperimentAgent
from auto_research.agents.intake import IntakeAgent, _c2c_evidence_brief, _c2c_static_bundle_validity
from auto_research.agents.plan import PlanAgent
from auto_research.agents.base import AgentContext
from auto_research.failure_log import build_c2c_feedback_bundle, load_c2c_feedback_bundle
from auto_research.artifacts import ArtifactManager
from auto_research.c2c import C2CAdapter, C2CPatchGuard, DEFAULT_C2C_PROXY_SCREEN, c2c_idea_novelty_report, collect_c2c_eval_smoke, default_c2c_ideas
from auto_research.direction_contracts import build_direction_contract, build_s1_direction_fingerprint, build_s1_evidence_quality_score
from auto_research.evidence_refs import resolve_s1_evidence_refs
from auto_research.judges import gate_s0
from auto_research.s2_planner_contracts import build_s2_candidate_pool, build_s2_planner_gate_report, build_s2_variant_scorecard
from auto_research.s2_feedback_policy import build_s2_adaptive_policy, build_s2_feedback_context, build_s2_score_adjustment_report
from auto_research.validators.s2_gate import S2GateValidator
from auto_research.code_patch import (
    CodePatchAgent,
    DynamicEditPolicy,
    FrozenPatchGuard,
    _code_patch_config,
    _codex_retryable_error_text,
    _normalize_activation_forward_probe_payload,
    _patch_failure_retryable,
    _runtime_smoke_gpu_attempts,
    _runtime_smoke_oom_retry_attempt,
)
from auto_research.llm import ModelClient
from auto_research.mineru import MinerUError, MinerUPdfClient
from auto_research.code_intake import retrieve_code_chunks
from auto_research.s0_enrichment import DeepSeekS0SemanticEnricher, S0SemanticEnrichmentError
from auto_research.orchestrator import Orchestrator
from auto_research.utils import sha256_file, write_json, write_yaml
from auto_research.workspace import init_workspace
from auto_research.cli import _run_c2c_command, _smoke_c2c_command


def _torch_available() -> bool:
    try:
        __import__("torch")
    except Exception:
        return False
    return True


def _transformers_available() -> bool:
    try:
        __import__("transformers")
    except Exception:
        return False
    return True


def _base_config(tmp_path: Path, *, simulate: bool = True) -> dict:
    return {
        "project": {"workspace_root": str(tmp_path), "target_venue": "TestConf", "language": "en"},
        "llm": {"provider": "openai", "use_real_api": False, "model": "mock"},
        "literature": {"download_pdfs": False, "request_timeout_seconds": 1, "max_papers": 0, "arxiv_max_results": 0},
        "experiment": {"simulate": simulate, "random_seeds": [42]},
        "writing": {"claim_verification": {"enabled": True, "min_pass_rate": 0.8}, "require_compile": False},
        "review": {"pass_threshold": 7.0, "max_iterations": 1},
        "orchestration": {"judge_max_retries": 1, "auto_mode": True},
    }


def _fake_c2c_repo(tmp_path: Path) -> Path:
    root = tmp_path / "C2C"
    for rel in [
        "rosetta/model",
        "script/train",
        "script/evaluation",
        "recipe/train_recipe",
        "recipe/eval_recipe",
        "local/final_results/route1_alignment_v22/small_loop_summary",
        "local/final_results/demo/mmlu-redux",
        "local/tmp/train_recipes/route1_alignment_v22",
        "local/tmp/eval_configs/route1_alignment_v22",
        "test",
        "wandb/offline-run",
        "local/checkpoints/demo",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    for name in ["README.md", "RUNBOOK.md", "C2C_跨Tokenizer柔性对齐改进方向研究备忘.md"]:
        (root / name).write_text(f"# {name}\nC2C cross-tokenizer cache communication.\n", encoding="utf-8")
    (root / "local/final_results/EXPERIMENT_RECORD.md").write_text("E0 baseline\nE20 v2.2 token_mlp\n", encoding="utf-8")
    (root / "rosetta/model/aligner.py").write_text("VALUE = 'aligner'\n", encoding="utf-8")
    (root / "rosetta/model/projector.py").write_text("VALUE = 'projector'\n", encoding="utf-8")
    (root / "rosetta/model/wrapper.py").write_text("VALUE = 'wrapper'\n", encoding="utf-8")
    (root / "script/train/SFT_train.py").write_text("print('train')\n", encoding="utf-8")
    (root / "script/evaluation/unified_evaluator.py").write_text("print('eval')\n", encoding="utf-8")
    (root / "test/test_aligner_span_overlap.py").write_text("def test_span():\n    assert True\n", encoding="utf-8")
    (root / "recipe/train_recipe/C2C_0.6+0.5.json").write_text(
        json.dumps({"output": {}, "data": {"kwargs": {}}, "training": {}, "model": {}}),
        encoding="utf-8",
    )
    (root / "recipe/eval_recipe/unified_eval.yaml").write_text(
        yaml.safe_dump({"model": {"rosetta_config": {}}, "output": {}, "eval": {"dataset": "mmlu-redux"}}),
        encoding="utf-8",
    )
    scores = root / "local/final_results/route1_alignment_v22/small_loop_summary/route1_v22_small_loop_scores.csv"
    scores.write_text(
        "\n".join(
            [
                "method,receiver,sharer,alignment_strategy,confidence_gate,train_samples,final_train_loss,mid_eval_loss,final_eval_loss,mmlu_redux,ai2_arc_challenge,openbookqa,mean,delta_mean_vs_v21_entropy050",
                "v2.2_token_mlp_entropy050,Qwen,Tiny,soft_span_overlap_v2,token_mlp,2048,0.37,0.17,0.16,47.07,54.78,50.60,50.82,1.05",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "model": "Rosetta",
        "dataset": "mmlu-redux",
        "answer_method": "generate",
        "overall_accuracy": 0.4707,
    }
    (root / "local/final_results/demo/mmlu-redux/Rosetta_mmlu-redux_generate_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (root / "wandb/offline-run/skip.txt").write_text("skip", encoding="utf-8")
    (root / "local/checkpoints/demo/model.pth").write_text("skip", encoding="utf-8")
    return root


def _fake_git_c2c_repo(tmp_path: Path) -> Path:
    repo = _fake_c2c_repo(tmp_path)
    (repo / ".git").mkdir(parents=True, exist_ok=True)
    return repo


def _s1_codex_direction_payload(
    *,
    code_chunk_id: str = "code:rosetta/model/aligner.py",
    used_shared_memory_refs: list[str] | None = None,
) -> dict:
    used_shared_memory_refs = list(used_shared_memory_refs or [])
    return {
        "schema_version": "c2c_s1_codex_direction_v1",
        "status": "ok",
        "used_shared_memory_refs": used_shared_memory_refs,
        "evidence_requests": [
            {
                "query": "utility cache routing implementation surface",
                "source_type": "code",
                "desired_evidence": "implementation",
                "why_needed": "S2 needs a bounded place to turn the direction into a patch.",
            },
            {
                "query": "paper evidence for utility-controlled cache transfer",
                "source_type": "paper",
                "desired_evidence": "support",
                "why_needed": "S1 must ground the mechanism direction in at least two paper refs.",
            },
            {
                "query": "reviewer counterevidence for cache routing collapse",
                "source_type": "rebuttal",
                "desired_evidence": "counterevidence",
                "why_needed": "S1 must inspect at least one risk before sending the direction to S2.",
            }
        ],
        "evidence_bundle": {
            "items": [
                {
                    "source_path": "intake/c2c/paper_chunks.jsonl",
                    "source_label": "paper:cache_utility_signal",
                    "source_type": "paper",
                    "summary": "Prior transfer methods benefit from utility-aware cache selection.",
                    "supports": ["utility_predicted_cache_routing"],
                    "risks": [],
                },
                {
                    "source_path": "intake/c2c/paper_chunks.jsonl",
                    "source_label": "paper:coverage_preserving_transfer",
                    "source_type": "paper",
                    "summary": "Coverage-preserving transfer avoids regressions from over-filtered cache states.",
                    "supports": ["utility_predicted_cache_routing"],
                    "risks": [],
                },
                {
                    "source_path": "intake/c2c/rebuttal_chunks.jsonl",
                    "source_label": "rebuttal:coverage_collapse",
                    "source_type": "rebuttal",
                    "summary": "Reviewer concerns emphasize OOD failure and dataset-level coverage collapse.",
                    "supports": [],
                    "risks": ["coverage_collapse"],
                },
                {
                    "chunk_id": code_chunk_id,
                    "source_path": "rosetta/model/aligner.py",
                    "source_type": "code",
                    "summary": "Alignment and cache transfer surfaces are localized in rosetta/model files.",
                    "supports": ["utility_predicted_cache_routing"],
                    "risks": [],
                },
                {
                    "source_path": "rosetta/model/projector.py",
                    "source_label": "rosetta/model/projector.py",
                    "source_type": "code",
                    "summary": "Projector code is a second bounded cache transfer surface for S2.",
                    "supports": ["utility_predicted_cache_routing"],
                    "risks": [],
                },
                {
                    "source_path": "intake/c2c/negative_result_memory.json",
                    "source_label": "feedback:coverage_collapse",
                    "source_type": "failure_feedback",
                    "summary": "Prior hard-gate style changes risk all-dataset collapse.",
                    "supports": [],
                    "risks": ["hard_gate_stack"],
                },
            ]
        },
        "direction_decision": {
            "direction_id": "utility_predicted_cache_routing",
            "mechanism_direction": "Utility Predicted Cache Routing",
            "mechanism_type": "utility_predicted_cache_routing",
            "core_hypothesis": "Predict downstream utility for transferred cache states and let S2 explore soft routing mechanisms that preserve baseline coverage.",
            "allowed_variants": ["soft residual utility scaling", "coverage-preserving utility modulation"],
            "forbidden_patterns": ["extra hard accept/reject gate", "evaluator changes"],
            "target_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
            "failure_focus": ["dataset-level coverage collapse", "mmlu-redux regression"],
            "expected_files": ["rosetta/model/aligner.py", "rosetta/model/projector.py", "rosetta/model/wrapper.py"],
            "verification_commands": ["py_compile", "small2048_train", "three_dataset_eval"],
            "rationale": "The direction is mechanism-level and leaves concrete patch variants to S2.",
            "used_shared_memory_refs": used_shared_memory_refs,
        },
        "selected_ideas": [
            {
                "id": "utility_predicted_cache_routing",
                "title": "Utility Predicted Cache Routing",
                "selected": True,
                "hypothesis": "Predict downstream utility for transferred cache states and route them without reducing baseline transfer coverage.",
                "novelty_score": 7,
                "feasibility_score": 7,
                "mechanism_type": "utility_predicted_cache_routing",
                "description": "High-level S1 direction only; S2 will generate concrete implementation candidates.",
                "motivation": "Baseline transfer lacks a downstream utility signal and previous failures warn against hard gating.",
                "reviewer_risk_response": "Track transfer coverage and per-dataset regressions; forbid evaluator edits and hard-gate stacking.",
                "expected_files": ["rosetta/model/aligner.py", "rosetta/model/projector.py", "rosetta/model/wrapper.py"],
                "verification_commands": ["py_compile", "small2048_train", "three_dataset_eval"],
                "evidence_refs": [
                    {"source_type": "paper", "source_path": "intake/c2c/paper_chunks.jsonl", "source_label": "paper:cache_utility_signal", "claim": "utility signal supports cache routing"},
                    {"source_type": "paper", "source_path": "intake/c2c/paper_chunks.jsonl", "source_label": "paper:coverage_preserving_transfer", "claim": "coverage preservation is required for safe transfer"},
                ],
                "counterevidence_refs": [
                    {
                        "source_type": "failure_feedback",
                        "source_path": "intake/c2c/negative_result_memory.json",
                        "source_label": "feedback:coverage_collapse",
                        "claim": "avoid hard gate collapse",
                    }
                ],
                "code_refs": [
                    {"source_type": "code", "source_label": "rosetta/model/aligner.py", "claim": "alignment/cache routing surface"},
                    {"source_type": "code", "source_label": "rosetta/model/projector.py", "claim": "projector/cache transfer surface"},
                ],
                "s1_allowed_variants": ["soft residual utility scaling", "coverage-preserving utility modulation"],
                "s1_forbidden_patterns": ["extra hard accept/reject gate", "evaluator changes"],
                "used_shared_memory_refs": used_shared_memory_refs,
            }
        ],
        "negative_constraints": {
            "reviewer_concerns": ["failure_modes_ood", "coverage collapse"],
            "forbidden_idea_ids": ["hard_gate_stack"],
            "forbidden_patterns": ["extra hard accept/reject gate", "evaluator changes"],
            "failure_feedback_rules": ["Use method-level failures only; ignore S2.5 coding noise in S1."],
            "used_shared_memory_refs": used_shared_memory_refs,
        },
        "decision_chain": {
            "evidence": ["paper:cache_utility_signal", "paper:coverage_preserving_transfer", code_chunk_id, "rosetta/model/projector.py"],
            "counterevidence": ["feedback:coverage_collapse"],
            "conclusion": "Use utility-predicted cache routing as the S1 direction and let S2 choose concrete variants.",
            "used_shared_memory_refs": used_shared_memory_refs,
        },
    }


def _write_direction_and_variant_gate_artifacts(project: Path, *, direction_id: str = "utility_predicted_cache_routing") -> None:
    (project / "literature").mkdir(parents=True, exist_ok=True)
    (project / "plan").mkdir(parents=True, exist_ok=True)
    direction = {
        "schema_version": "auto_research_direction_v1",
        "direction_id": direction_id,
        "title": "Utility Predicted Cache Routing",
        "mechanism_type": "utility_predicted_cache_routing",
        "mechanism_axis": "routing",
        "integration_point": "wrapper",
        "control_signal": "utility",
        "hypothesis": "Predict utility for transferred cache states.",
        "why_baseline_fails": "The baseline lacks downstream utility control.",
        "expected_metric_signature": {"primary_metric": "three_dataset_mean", "expected_direction": "increase"},
        "required_evidence_refs": [{"source_type": "code", "source_label": "rosetta/model/wrapper.py", "claim": "surface"}],
        "counterevidence_refs": [{"source_type": "failure_feedback", "source_label": "risk", "claim": "avoid hard gates"}],
        "implementation_surface_refs": [{"source_type": "code", "source_label": "rosetta/model/wrapper.py", "claim": "surface"}],
        "known_negative_memory_refs": [],
        "go_to_s2_conditions": ["evidence resolved"],
        "return_to_s1_conditions": ["budget exhausted"],
        "expected_files": ["rosetta/model/wrapper.py"],
        "verification_commands": ["py_compile"],
        "used_shared_memory_refs": [],
    }
    variant = {
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
    }
    (project / "literature" / "direction.json").write_text(json.dumps(direction), encoding="utf-8")
    (project / "plan" / "planner_decision.json").write_text(
        json.dumps(
            {
                "schema_version": "auto_research_planner_decision_v1",
                "direction_id": direction_id,
                "planner_summary": "Select wrapper utility variant.",
                "planning_mode": "same_direction_variant",
                "used_shared_memory_refs": [],
                "next_variant": variant,
            }
        ),
        encoding="utf-8",
    )
    (project / "plan" / "variant_contract.json").write_text(
        json.dumps(
            {
                "schema_version": "auto_research_variant_contract_v1",
                "direction_id": direction_id,
                "variant_id": variant["id"],
                "title": variant["title"],
                "mode": "regular",
                "variant_fingerprint": variant["variant_fingerprint"],
                "mechanism_axis": "routing",
                "integration_point": "wrapper",
                "control_signal": "utility",
                "hypothesis": direction["hypothesis"],
                "why_next": "Wrapper utility routing.",
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
        ),
        encoding="utf-8",
    )
    (project / "plan" / "variant_fingerprint.json").write_text(
        json.dumps(
            {
                "schema_version": "auto_research_variant_fingerprint_v1",
                "direction_id": direction_id,
                "variant_id": variant["id"],
                "variant_fingerprint": variant["variant_fingerprint"],
                "mechanism_axis": "routing",
                "integration_point": "wrapper",
                "control_signal": "utility",
                "history_fingerprints": [],
                "is_repeat": False,
                "mode": "regular",
            }
        ),
        encoding="utf-8",
    )
    contract = json.loads((project / "plan" / "variant_contract.json").read_text(encoding="utf-8"))
    fingerprint = json.loads((project / "plan" / "variant_fingerprint.json").read_text(encoding="utf-8"))
    candidate_pool = build_s2_candidate_pool(direction=direction, candidates=[variant], source="test_fixture")
    feedback_context = build_s2_feedback_context(project_root=project, direction=direction, config={})
    adaptive_policy = build_s2_adaptive_policy(feedback_context, {})
    scorecard = build_s2_variant_scorecard(
        direction=direction,
        candidate_pool=candidate_pool,
        selected_variant=variant,
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
    planner_gate = build_s2_planner_gate_report(
        direction=direction,
        candidate_pool=candidate_pool,
        scorecard=scorecard,
        next_variant=variant,
        variant_contract=contract,
        variant_fingerprint=fingerprint,
        adaptive_policy=adaptive_policy,
        score_adjustment_report=score_adjustment_report,
        config={},
    )
    (project / "plan" / "s2_planner").mkdir(parents=True, exist_ok=True)
    (project / "plan" / "next_variant.json").write_text((project / "plan" / "planner_decision.json").read_text(encoding="utf-8"), encoding="utf-8")
    (project / "plan" / "s2_planner" / "candidate_pool.json").write_text(json.dumps(candidate_pool), encoding="utf-8")
    (project / "plan" / "s2_planner" / "feedback_context.json").write_text(json.dumps(feedback_context), encoding="utf-8")
    (project / "plan" / "s2_planner" / "adaptive_policy.json").write_text(json.dumps(adaptive_policy), encoding="utf-8")
    (project / "plan" / "s2_planner" / "variant_scorecard.json").write_text(json.dumps(scorecard), encoding="utf-8")
    (project / "plan" / "s2_planner" / "score_adjustment_report.json").write_text(json.dumps(score_adjustment_report), encoding="utf-8")
    (project / "plan" / "s2_planner" / "next_variant.json").write_text(json.dumps(variant), encoding="utf-8")
    (project / "plan" / "s2_planner" / "planner_gate_report.json").write_text(json.dumps(planner_gate), encoding="utf-8")


def _write_minimal_s1_ref_catalog(project_root: Path) -> None:
    intake = project_root / "intake" / "c2c"
    intake.mkdir(parents=True, exist_ok=True)
    (project_root / "literature" / "c2c").mkdir(parents=True, exist_ok=True)
    chunk_index = {
        "schema_version": "c2c_full_chunk_index_v1",
        "counts": {"paper": 2, "rebuttal": 1, "code": 2, "total": 6},
        "entries": [
            {
                "chunk_id": "code:rosetta/model/aligner.py",
                "source_type": "code",
                "source_path": "rosetta/model/aligner.py",
                "section": "file",
                "keywords": ["aligner"],
                "text_preview": "aligner",
                "path": "rosetta/model/aligner.py",
                "symbol": "aligner",
            },
            {
                "chunk_id": "code:rosetta/model/projector.py",
                "source_type": "code",
                "source_path": "rosetta/model/projector.py",
                "section": "file",
                "keywords": ["projector"],
                "text_preview": "projector",
                "path": "rosetta/model/projector.py",
                "symbol": "projector",
            },
            {
                "chunk_id": "feedback:coverage_collapse",
                "source_type": "failure_feedback",
                "source_path": "intake/c2c/negative_result_memory.json",
                "section": "failure",
                "keywords": ["coverage"],
                "text_preview": "coverage collapse",
            },
            {
                "chunk_id": "paper:cache_routing",
                "source_type": "paper",
                "source_path": "intake/c2c/paper_chunks.jsonl",
                "section": "method",
                "keywords": ["cache"],
                "text_preview": "cache routing",
            },
            {
                "chunk_id": "paper:coverage_preserving_transfer",
                "source_type": "paper",
                "source_path": "intake/c2c/paper_chunks.jsonl",
                "section": "method",
                "keywords": ["coverage"],
                "text_preview": "coverage-preserving transfer",
            },
            {
                "chunk_id": "rebuttal:coverage_collapse",
                "source_type": "rebuttal",
                "source_path": "intake/c2c/rebuttal_chunks.jsonl",
                "section": "concern",
                "keywords": ["collapse"],
                "text_preview": "coverage collapse concern",
            },
        ],
    }
    (intake / "chunk_index.json").write_text(json.dumps(chunk_index), encoding="utf-8")
    (intake / "code_file_manifest.json").write_text(json.dumps({"files": [{"path": "rosetta/model/aligner.py"}, {"path": "rosetta/model/projector.py"}, {"path": "rosetta/model/wrapper.py"}]}), encoding="utf-8")
    (intake / "code_symbols.jsonl").write_text(
        json.dumps({"symbol": "aligner", "path": "rosetta/model/aligner.py"}) + "\n" + json.dumps({"symbol": "projector", "path": "rosetta/model/projector.py"}) + "\n",
        encoding="utf-8",
    )
    (intake / "code_chunks.jsonl").write_text(
        json.dumps({"chunk_id": "code:rosetta/model/aligner.py", "path": "rosetta/model/aligner.py", "symbol": "aligner"})
        + "\n"
        + json.dumps({"chunk_id": "code:rosetta/model/projector.py", "path": "rosetta/model/projector.py", "symbol": "projector"})
        + "\n",
        encoding="utf-8",
    )
    (intake / "negative_result_memory.json").write_text(json.dumps({"blocked": ["coverage"]}), encoding="utf-8")
    (intake / "paper_chunks.jsonl").write_text(
        json.dumps({"chunk_id": "paper:cache_routing", "source_path": "paper.md"})
        + "\n"
        + json.dumps({"chunk_id": "paper:coverage_preserving_transfer", "source_path": "paper.md"})
        + "\n",
        encoding="utf-8",
    )
    (intake / "rebuttal_chunks.jsonl").write_text(json.dumps({"chunk_id": "rebuttal:coverage_collapse", "source_path": "rebuttal.md"}) + "\n", encoding="utf-8")


def _quality_score_for_s1_payload(project_root: Path, payload: dict, *, novelty_score: float = 0.68) -> dict:
    report = resolve_s1_evidence_refs(project_root, payload, mode="c2c")
    direction = build_direction_contract(payload, mode="c2c", used_shared_memory_refs=payload.get("used_shared_memory_refs") or [])
    fingerprint = build_s1_direction_fingerprint(direction, project_root=project_root)
    return build_s1_evidence_quality_score(
        direction,
        payload=payload,
        evidence_bundle=payload.get("evidence_bundle"),
        evidence_ref_report=report,
        novelty_audit={
            "schema_version": "auto_research_novelty_audit_v1",
            "direction_id": direction.get("direction_id"),
            "status": "ok",
            "enabled": True,
            "passed": novelty_score >= 0.6,
            "threshold": 0.6,
            "latest": {"status": "ok", "passed": novelty_score >= 0.6, "audit": {"novelty_score": novelty_score}},
            "audits": [],
        },
        direction_fingerprint=fingerprint,
        shared_memory_checked=True,
    )


def test_c2c_s1_evidence_quality_score_passes_with_required_coverage(tmp_path: Path) -> None:
    project_root = tmp_path / "p_quality_pass"
    project_root.mkdir()
    _write_minimal_s1_ref_catalog(project_root)

    score = _quality_score_for_s1_payload(project_root, _s1_codex_direction_payload())

    assert score["gate"] == "pass"
    assert score["support_coverage"]["paper"] >= 2
    assert score["support_coverage"]["code"] >= 2
    assert score["counterevidence"]["count"] >= 1
    assert score["implementation_surface_coverage"] >= 0.6
    assert score["novelty_score"] >= 0.6


def test_c2c_s1_evidence_quality_counts_distinct_code_refs_in_same_file(tmp_path: Path) -> None:
    project_root = tmp_path / "p_quality_same_file_code_refs"
    project_root.mkdir()
    _write_minimal_s1_ref_catalog(project_root)
    code_chunks_path = project_root / "intake" / "c2c" / "code_chunks.jsonl"
    code_chunks_path.write_text(
        code_chunks_path.read_text(encoding="utf-8")
        + json.dumps({"chunk_id": "code:rosetta/model/wrapper.py:init", "path": "rosetta/model/wrapper.py", "symbol": "RosettaModel.__init__"})
        + "\n"
        + json.dumps({"chunk_id": "code:rosetta/model/wrapper.py:forward", "path": "rosetta/model/wrapper.py", "symbol": "RosettaModel.forward"})
        + "\n",
        encoding="utf-8",
    )
    payload = _s1_codex_direction_payload()
    wrapper_code_refs = [
        {"source_type": "code", "source_path": "rosetta/model/wrapper.py", "source_label": "rosetta/model/wrapper.py::RosettaModel.__init__", "claim": "wrapper init surface"},
        {"source_type": "code", "source_path": "rosetta/model/wrapper.py", "source_label": "rosetta/model/wrapper.py::RosettaModel.forward", "claim": "wrapper forward surface"},
    ]
    payload["evidence_bundle"]["items"] = [
        *[item for item in payload["evidence_bundle"]["items"] if item.get("source_type") != "code"],
        *wrapper_code_refs,
    ]
    payload["direction_decision"]["expected_files"] = ["rosetta/model/wrapper.py"]
    payload["selected_ideas"][0]["expected_files"] = ["rosetta/model/wrapper.py"]
    payload["selected_ideas"][0]["code_refs"] = wrapper_code_refs

    score = _quality_score_for_s1_payload(project_root, payload)

    assert score["gate"] == "pass"
    assert score["support_coverage"]["code"] == 2


@pytest.mark.parametrize(
    ("case_name", "mutate", "expected_rule"),
    [
        (
            "unresolved_ref",
            lambda payload: payload["selected_ideas"][0]["code_refs"].append({"source_type": "code", "source_label": "missing/file.py", "claim": "bad ref"}),
            "unresolved_ref_count",
        ),
        (
            "too_few_paper_refs",
            lambda payload: (
                payload["selected_ideas"][0].update({"evidence_refs": payload["selected_ideas"][0]["evidence_refs"][:1]}),
                payload["evidence_bundle"].update({"items": [item for item in payload["evidence_bundle"]["items"] if item.get("source_type") != "paper"] + [payload["evidence_bundle"]["items"][0]]}),
            ),
            "support_coverage.paper",
        ),
        (
            "too_few_code_refs",
            lambda payload: (
                payload["selected_ideas"][0].update({"code_refs": payload["selected_ideas"][0]["code_refs"][:1]}),
                payload["evidence_bundle"].update({"items": [item for item in payload["evidence_bundle"]["items"] if item.get("source_type") != "code" or item.get("source_path") == "rosetta/model/aligner.py"]}),
            ),
            "support_coverage.code",
        ),
        (
            "missing_counterevidence",
            lambda payload: (
                payload["selected_ideas"][0].update({"counterevidence_refs": []}),
                [item.update({"risks": []}) for item in payload["evidence_bundle"]["items"]],
            ),
            "counterevidence.resolved_count",
        ),
        (
            "low_surface_coverage",
            lambda payload: (
                payload["direction_decision"].update({"expected_files": ["rosetta/model/aligner.py", "rosetta/model/projector.py", "rosetta/model/wrapper.py", "rosetta/model/router.py", "rosetta/model/loss.py"]}),
                payload["selected_ideas"][0].update({"expected_files": ["rosetta/model/aligner.py", "rosetta/model/projector.py", "rosetta/model/wrapper.py", "rosetta/model/router.py", "rosetta/model/loss.py"]}),
            ),
            "implementation_surface_coverage",
        ),
        (
            "low_novelty",
            lambda payload: None,
            "novelty_score",
        ),
    ],
)
def test_c2c_s1_evidence_quality_score_fails_hard_rules(tmp_path: Path, case_name: str, mutate, expected_rule: str) -> None:
    project_root = tmp_path / f"p_quality_{case_name}"
    project_root.mkdir()
    _write_minimal_s1_ref_catalog(project_root)
    payload = json.loads(json.dumps(_s1_codex_direction_payload()))
    mutate(payload)

    score = _quality_score_for_s1_payload(project_root, payload, novelty_score=0.59 if case_name == "low_novelty" else 0.68)

    assert score["gate"] == "fail"
    assert expected_rule in score["failed_rules"]


def test_c2c_s1_merges_s0_semantic_enrichment_into_chunk_catalog(tmp_path: Path) -> None:
    project_root = tmp_path / "proj_semantic_s1"
    intake = project_root / "intake" / "c2c"
    cache = project_root / ".cache" / "auto_research" / "s0_semantic_enrichment" / "deepseek-v4-flash"
    intake.mkdir(parents=True)
    cache.mkdir(parents=True)
    paper_record = {
        "generated_at": "2026-06-03T00:00:00+00:00",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "prompt_version": "deepseek_s0_semantic_enrichment_v1",
        "cache_status": "miss",
        "fallback_used": False,
        "chunk": {"chunk_id": "paper:method", "source_type": "paper"},
        "enrichment": {
            "semantic_summary": "Paper says cache routing should preserve useful transferred states.",
            "mechanism_tags": ["cache routing", "coverage preservation"],
            "failure_modes": ["coverage collapse"],
            "retrieval_keywords": ["utility routing", "soft cache routing"],
        },
    }
    stale_code_record = {
        "generated_at": "2026-06-03T00:00:01+00:00",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "prompt_version": "deepseek_s0_semantic_enrichment_v1",
        "cache_status": "miss",
        "fallback_used": True,
        "chunk": {"chunk_id": "code:aligner", "source_type": "code"},
        "enrichment": {"semantic_summary": "Stale fallback summary.", "retrieval_keywords": ["stale"]},
    }
    code_record = {
        "generated_at": "2026-06-03T00:00:02+00:00",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "prompt_version": "deepseek_s0_code_semantic_enrichment_v2",
        "cache_status": "miss",
        "fallback_used": False,
        "chunk": {"chunk_id": "code:aligner", "source_type": "code"},
        "enrichment": {
            "semantic_summary": "Aligner is the bounded runtime patch surface for cache transfer and valid_mask handling.",
            "mechanism_tags": ["runtime", "valid_mask", "alignment"],
            "failure_modes": ["dtype/device mismatch"],
            "retrieval_keywords": ["aligner", "valid_mask", "cache transfer"],
        },
    }
    write_json(intake / "semantic_enrichment_sample.json", {"records": [paper_record, stale_code_record]})
    write_json(cache / "code.json", code_record)
    paper_chunks = [{"chunk_id": "paper:method", "source_type": "paper", "source_path": "paper.md", "section": "method", "keywords": ["cache"]}]
    rebuttal_chunks = []
    code_chunks = [{"chunk_id": "code:aligner", "source_type": "code", "source_path": "rosetta/model/aligner.py", "path": "rosetta/model/aligner.py", "keywords": ["aligner"]}]
    chunk_index = {
        "counts": {"paper": 1, "rebuttal": 0, "code": 1, "total": 2},
        "entries": [
            {"chunk_id": "paper:method", "source_type": "paper", "source_path": "paper.md", "section": "method", "keywords": ["cache"]},
            {"chunk_id": "code:aligner", "source_type": "code", "source_path": "rosetta/model/aligner.py", "section": "file", "keywords": ["aligner"]},
        ],
    }

    merged = literature_module._merge_s0_semantic_enrichment_for_s1(
        project_root,
        paper_chunks=paper_chunks,
        rebuttal_chunks=rebuttal_chunks,
        code_chunks=code_chunks,
        chunk_index=chunk_index,
        config={"intake": {"semantic_enrichment": {"model": "deepseek-v4-flash"}}},
    )

    assert merged["report"]["records_loaded"] == 2
    assert merged["report"]["chunks_enriched"] == {"paper": 1, "rebuttal": 0, "code": 1}
    code_entry = next(entry for entry in merged["chunk_index"]["entries"] if entry["chunk_id"] == "code:aligner")
    assert code_entry["semantic_summary"].startswith("Aligner is the bounded runtime patch surface")
    assert code_entry["semantic_enrichment"]["prompt_version"] == "deepseek_s0_code_semantic_enrichment_v2"
    prompt_catalog = literature_module._summarize_chunk_index_for_prompt(merged["chunk_index"])
    prompt_code_entry = next(entry for entry in prompt_catalog["entries"] if entry["chunk_id"] == "code:aligner")
    assert "valid_mask" in prompt_code_entry["retrieval_keywords"]
    assert prompt_code_entry["failure_modes"] == ["dtype/device mismatch"]


def test_init_c2c_creates_snapshot_and_config(monkeypatch, tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "paper.txt"
    ref_rebuttal = tmp_path / "rebuttal.md"
    ref_paper.write_text("paper text", encoding="utf-8")
    ref_rebuttal.write_text("review text", encoding="utf-8")
    config = _base_config(tmp_path / "workspace")
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)

    project_id = Orchestrator().init_c2c_project(
        "cross tokenizer cache",
        target_repo=source_repo,
        ref_paper=ref_paper,
        ref_rebuttal=ref_rebuttal,
        env_python=Path("/usr/bin/python3"),
        project_id="proj_c2c",
        simulate=True,
    )

    root = tmp_path / "workspace" / project_id
    project_config = yaml.safe_load((root / "meta/project_config.yaml").read_text(encoding="utf-8"))
    assert project_config["c2c"]["enabled"] is True
    assert (root / "external/c2c_snapshot/rosetta/model/aligner.py").exists()
    assert not (root / "external/c2c_snapshot/wandb").exists()
    assert not (root / "external/c2c_snapshot/local/checkpoints").exists()
    assert (root / "experiment/c2c/repo_snapshot_manifest.json").exists()
    manifest = json.loads((root / "experiment/c2c/repo_snapshot_manifest.json").read_text(encoding="utf-8"))
    assert "source_git_commit" in manifest


def test_run_c2c_command_prepares_three_iteration_project_with_s0_cache(monkeypatch, tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "paper.txt"
    ref_rebuttal = tmp_path / "rebuttal.md"
    ref_paper.write_text("paper text", encoding="utf-8")
    ref_rebuttal.write_text("review text", encoding="utf-8")
    config = _base_config(tmp_path / "workspace", simulate=False)
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(cli_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)

    orchestrator = Orchestrator()
    source_project = orchestrator.init_c2c_project(
        "source cache",
        target_repo=source_repo,
        ref_paper=ref_paper,
        ref_rebuttal=ref_rebuttal,
        env_python=Path("/usr/bin/python3"),
        project_id="source_cache_project",
        simulate=False,
    )
    source_root = tmp_path / "workspace" / source_project
    code_chunk = {
        "chunk_id": "code:rosetta/model/aligner.py::align",
        "path": "rosetta/model/aligner.py",
        "source_type": "code",
        "source_path": "rosetta/model/aligner.py",
        "section": "align",
        "text": "def align():\n    return True\n",
    }
    bundle = {
        "schema_version": "c2c_static_intake_bundle_v1",
        "metadata": [],
        "reference_result": {},
        "paper_full_manifest": [],
        "repo_manifest": {
            "core_files": [
                {
                    "path": "rosetta/model/aligner.py",
                    "sha256": sha256_file(source_root / "external/c2c_snapshot/rosetta/model/aligner.py"),
                }
            ]
        },
        "historical_results": {"results": []},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "repo_card": {},
        "paper_cards": [],
        "paper_chunks": [{"chunk_id": "paper:1", "source_type": "paper", "source_path": "paper.md", "text": "paper"}],
        "bibliography_cards": [],
        "rebuttal_matrix": {},
        "rebuttal_chunks": [{"chunk_id": "rebuttal:1", "source_type": "rebuttal", "source_path": "rebuttal.md", "text": "review"}],
        "code_cards": [],
        "code_file_manifest": {"files": []},
        "code_symbols": [],
        "code_chunks": [code_chunk],
        "code_edges": [],
        "code_repo_map": {},
        "code_intake_report": {},
        "implementation_surface_map": {},
        "code_retrieval_index": {},
        "cache_summary": {},
        "chunk_index": {
            "counts": {"paper": 1, "rebuttal": 1, "code": 1, "total": 3},
            "entries": [
                {"chunk_id": "paper:1", "source_type": "paper", "source_path": "paper.md", "text": "paper"},
                {"chunk_id": "rebuttal:1", "source_type": "rebuttal", "source_path": "rebuttal.md", "text": "review"},
                code_chunk,
            ],
        },
        "result_ledger_csv": "id,mean\n",
        "negative_memory": {},
        "retrieval_plan": {},
        "followup_bundle": {},
        "evidence_brief": {"schema_version": "c2c_evidence_brief_v1"},
    }
    write_json(source_root / "intake/c2c/static_bundle.json", bundle)
    sidecar = source_root / "references/c2c/ref_paper/demo/paper_full.md"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("# Cached Paper\n\nMethod.\n", encoding="utf-8")
    bundle["paper_full_manifest"] = [
        {
            "paper_id": "demo",
            "paper_full_md_path": "references/c2c/ref_paper/demo/paper_full.md",
            "parser_artifacts": ["references/c2c/ref_paper/demo/paper_full.md"],
        }
    ]
    write_json(source_root / "intake/c2c/static_bundle.json", bundle)

    args = SimpleNamespace(
        topic="cross tokenizer cache communication",
        project_id="new_run_project",
        target_repo=str(source_repo),
        ref_paper=str(ref_paper),
        ref_rebuttal=str(ref_rebuttal),
        env_python="/usr/bin/python3",
        max_iterations=3,
        stop_after_stage="S3_experiment",
        simulate=False,
        hitl=False,
        no_s0_cache=False,
        s0_cache_project=source_project,
        s0_cache_path=None,
        s0_force_refresh=False,
        prepare_only=True,
    )

    result = _run_c2c_command(args, orchestrator)

    new_root = tmp_path / "workspace/new_run_project"
    project_config = yaml.safe_load((new_root / "meta/project_config.yaml").read_text(encoding="utf-8"))
    registry = yaml.safe_load((new_root / "meta/registry.yaml").read_text(encoding="utf-8"))
    restored_bundle = json.loads((new_root / "intake/c2c/static_bundle.json").read_text(encoding="utf-8"))
    assert result["status"] == "prepared"
    assert project_config["c2c"]["enabled"] is True
    assert project_config["orchestration"]["auto_mode"] is True
    assert project_config["orchestration"]["stop_after_stage"] == "S3_experiment"
    assert project_config["review"]["max_iterations"] == 3
    assert registry["max_iterations"] == 3
    assert restored_bundle["project_id"] == "new_run_project"
    assert result["s0_cache"]["status"] == "restored"
    assert result["s0_cache"]["sidecars"]["copied_count"] == 1
    assert (new_root / "references/c2c/ref_paper/demo/paper_full.md").exists()


def test_smoke_c2c_command_stops_on_readiness_failure(monkeypatch, tmp_path: Path) -> None:
    project = tmp_path / "workspace" / "proj_smoke_fail"
    _write_smoke_registry(project, current_stage="S0_intake", status="initialized")
    write_yaml(project / "meta" / "project_config.yaml", {"c2c": {"enabled": True}, "experiment": {"simulate": False}})
    monkeypatch.setattr(cli_module, "load_project_config", lambda project_root: {"c2c": {"enabled": True}, "experiment": {"simulate": False}})

    class FakeOrchestrator:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def _project_root(self, project_id: str) -> Path:
            assert project_id == "proj_smoke_fail"
            return project

        def doctor_c2c(self, project_id: str) -> dict:
            self.calls.append("doctor-c2c")
            readiness = _smoke_readiness(project, gate="fail", blocking=["env_python_executable"])
            write_json(project / "meta" / "c2c_e2e_readiness_report.json", readiness)
            write_json(project / "meta" / "c2c_runtime_health_report.json", {"project_id": project.name})
            return {"status": "fail", "readiness_report": readiness}

    def unexpected_run(*args, **kwargs):
        raise AssertionError("smoke-c2c must not start run-c2c when readiness fails")

    monkeypatch.setattr(cli_module, "_run_c2c_command", unexpected_run)
    result = _smoke_c2c_command(SimpleNamespace(project_id="proj_smoke_fail", from_stage="S3_experiment"), FakeOrchestrator())

    assert result["status"] == "readiness_failed"
    assert result["steps"] == [{"name": "doctor-c2c", "status": "fail"}]
    record = json.loads((project / "meta/c2c_real_smoke_record.json").read_text(encoding="utf-8"))
    assert record["readiness_gate"] == "fail"
    assert "env_python_executable" in record["blocking_reasons"]


def test_smoke_c2c_command_runs_real_smoke_sequence(monkeypatch, tmp_path: Path) -> None:
    project = tmp_path / "workspace" / "proj_smoke"
    _write_smoke_registry(project, current_stage="S3_experiment", status="completed")
    write_yaml(project / "meta" / "project_config.yaml", {"c2c": {"enabled": True}, "experiment": {"simulate": False}})
    monkeypatch.setattr(cli_module, "load_project_config", lambda project_root: {"c2c": {"enabled": True}, "experiment": {"simulate": False}})
    run_args_seen: dict[str, object] = {}

    class FakeOrchestrator:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def _project_root(self, project_id: str) -> Path:
            assert project_id == "proj_smoke"
            return project

        def doctor_c2c(self, project_id: str) -> dict:
            self.calls.append("doctor-c2c")
            readiness = _smoke_readiness(project, gate="pass", blocking=[])
            write_json(project / "meta" / "c2c_e2e_readiness_report.json", readiness)
            return {"status": "pass", "readiness_report": readiness}

        def audit_c2c(self, project_id: str, *, scope: str | None = None) -> dict:
            self.calls.append("audit-c2c")
            audit = _smoke_audit(project, gate="pass")
            write_json(project / "meta" / "c2c_artifact_audit_report.json", audit)
            return {"status": "pass", "artifact_audit_report": audit}

        def replay_c2c(self, project_id: str, *, from_stage: str = "S3_experiment") -> dict:
            self.calls.append(f"replay-c2c:{from_stage}")
            replay = _smoke_replay(project, status="match")
            write_json(project / "meta" / "c2c_replay_result.json", replay)
            return {"status": "match", "replay_result": replay}

        def report(self, project_id: str) -> dict:
            self.calls.append("report")
            return {"project_id": project_id, "e2e": {"real_smoke_record": {"last_stage": "S3_experiment"}}}

    orchestrator = FakeOrchestrator()

    def fake_run_c2c(args, _orchestrator):
        run_args_seen.update(vars(args))
        write_json(project / "meta" / "c2c_e2e_run_manifest.json", _smoke_manifest(project, final_status="completed"))
        return {"status": "completed", "project_id": args.project_id, "run_overrides": {"max_iterations": args.max_iterations}}

    monkeypatch.setattr(cli_module, "_run_c2c_command", fake_run_c2c)

    parsed = cli_module.build_parser().parse_args(["smoke-c2c", "--project-id", "proj_smoke"])
    assert parsed.command == "smoke-c2c"
    result = _smoke_c2c_command(SimpleNamespace(project_id="proj_smoke", from_stage="S3_experiment"), orchestrator)

    assert result["status"] == "passed"
    assert [step["name"] for step in result["steps"]] == ["doctor-c2c", "run-c2c", "audit-c2c", "replay-c2c", "report --json"]
    assert orchestrator.calls == ["doctor-c2c", "audit-c2c", "replay-c2c:S3_experiment", "report"]
    assert run_args_seen["max_iterations"] == 1
    assert run_args_seen["stop_after_stage"] == "S3_experiment"
    assert run_args_seen["simulate"] is False
    assert run_args_seen["hitl"] is False
    record = json.loads((project / "meta/c2c_real_smoke_record.json").read_text(encoding="utf-8"))
    assert record["readiness_gate"] == "pass"
    assert record["run_manifest_final_status"] == "completed"
    assert record["artifact_audit_gate"] == "pass"
    assert record["replay_status"] == "match"


def _write_smoke_registry(project: Path, *, current_stage: str, status: str) -> None:
    write_yaml(
        project / "meta" / "registry.yaml",
        {
            "project_id": project.name,
            "research_topic": "c2c smoke",
            "current_stage": current_stage,
            "iteration": 1,
            "status": status,
            "stages": {},
        },
    )


def _smoke_readiness(project: Path, *, gate: str, blocking: list[str]) -> dict:
    return {
        "schema_version": "c2c_e2e_readiness_report_v1",
        "project_id": project.name,
        "mode": "real",
        "gate": gate,
        "checks": {"env_python_executable": gate != "fail", "real_execution_hooks_ready": gate != "fail"},
        "warnings": [],
        "blocking_reasons": blocking,
        "recommended_action": "run_c2c" if gate != "fail" else "fix_environment",
    }


def _smoke_manifest(project: Path, *, final_status: str) -> dict:
    return {
        "schema_version": "c2c_e2e_run_manifest_v1",
        "project_id": project.name,
        "mode": "real",
        "command": {"name": "smoke-c2c"},
        "stage_boundaries": {"S3_experiment": {"status": final_status}},
        "final_status": final_status,
    }


def _smoke_audit(project: Path, *, gate: str) -> dict:
    return {
        "schema_version": "c2c_artifact_audit_report_v1",
        "project_id": project.name,
        "gate": gate,
        "audit_scope": "completed",
        "expected_stages": [],
        "skipped_stages": [],
        "summary": {"checked_artifacts": 1, "missing": 0, "schema_failures": 0, "missing_manifest_hash": 0, "hash_mismatches": 0, "stale_artifacts": 0},
        "by_stage": {},
        "blocking_reasons": [],
    }


def _smoke_replay(project: Path, *, status: str) -> dict:
    return {
        "schema_version": "c2c_replay_result_v1",
        "project_id": project.name,
        "status": status,
        "replayed_decisions": {},
        "expected_decisions": {},
        "mismatches": [],
    }


def test_c2c_importers_parse_refs_and_historical_results(tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "paper.txt"
    ref_rebuttal = tmp_path / "rebuttal.json"
    ref_paper.write_text(
        "Abstract\npaper method core for KV cache tokenizer communication.\n"
        "Method\nThe alignment method shares cache states across tokenizer boundaries.\n"
        "Experiments\nThe benchmark reports baseline accuracy and ablation results.\n"
        "References\n[1] unrelated prior work.\n",
        encoding="utf-8",
    )
    ref_rebuttal.write_text(
        json.dumps({"review": "needs stronger tokenizer mismatch evidence and baseline fairness discussion"}),
        encoding="utf-8",
    )
    paths = init_workspace(_base_config(tmp_path / "workspace"), "topic", project_id="proj", simulate=True)
    config_patch = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(source_repo),
            "ref_paper": str(ref_paper),
            "ref_rebuttal": str(ref_rebuttal),
            "env_python": "/usr/bin/python3",
        }
    }
    (paths.root / "meta/project_config.yaml").write_text(yaml.safe_dump(config_patch), encoding="utf-8")

    adapter = C2CAdapter(paths.root, config_patch)
    refs = adapter.import_reference_materials()
    history = adapter.import_historical_results()
    baseline = adapter.baseline_evidence(history)
    repo_manifest = adapter.build_repo_manifest()
    repo_card = adapter.build_repo_card(repo_manifest, history)
    paper_cards = adapter.build_paper_cards(refs["cards"])
    paper_chunks = adapter.build_paper_chunks(refs["cards"])
    bibliography = adapter.build_bibliography_cards(refs["cards"])
    rebuttal_matrix = adapter.build_rebuttal_concern_matrix(refs["cards"])
    rebuttal_chunks = adapter.build_rebuttal_chunks(refs["cards"])
    code_cards = adapter.build_code_cards(repo_manifest)
    code_intake = adapter.build_code_intake()
    code_chunks = code_intake.chunks
    chunk_index = adapter.build_chunk_index(
        paper_chunks=paper_chunks,
        rebuttal_chunks=rebuttal_chunks,
        code_chunks=code_chunks,
    )
    negative_memory = adapter.build_negative_result_memory(history, baseline)
    retrieval_plan = adapter.build_research_retrieval_plan(
        topic="cross tokenizer cache",
        repo_card=repo_card,
        paper_cards=paper_cards,
        paper_chunks=paper_chunks,
        rebuttal_matrix=rebuttal_matrix,
        rebuttal_chunks=rebuttal_chunks,
        code_cards=code_cards,
        code_chunks=code_chunks,
        negative_memory=negative_memory,
        baseline=baseline,
    )
    result_ledger = adapter.build_result_ledger_csv(history, baseline)

    assert refs["status"] == "ok"
    assert len(refs["cards"]) == 2
    assert history["counts"]["small_loop_rows"] == 1
    assert baseline["mean"] == 50.06
    assert repo_card["baseline"]["name"] == "paper_original_rosetta_fuser"
    assert len(paper_cards) == 1
    assert paper_chunks
    assert paper_chunks[0]["keywords"]
    assert "References" not in paper_chunks[-1]["text"]
    assert bibliography[0]["entry_count"] == 1
    assert rebuttal_chunks
    assert rebuttal_chunks[0]["keywords"]
    assert code_cards
    assert code_chunks
    assert code_chunks[0]["keywords"]
    assert code_intake.file_manifest["files"]
    assert code_intake.repo_map["counts"]["chunks"] == len(code_chunks)
    assert code_intake.repo_map["counts"]["symbols"] == len(code_intake.symbols)
    assert code_intake.report["counts"]["chunks"] == len(code_chunks)
    assert "surfaces" in code_intake.surface_map
    assert code_intake.retrieval_index["default_queries"]
    assert any(chunk["edit_surface"] in {"allowed", "allowed_prefix"} for chunk in code_chunks)
    assert any(edge["edge_type"] == "same_file_neighbor" for edge in code_intake.edges)
    assert chunk_index["counts"]["paper"] == len(paper_chunks)
    assert chunk_index["counts"]["rebuttal"] == len(rebuttal_chunks)
    assert chunk_index["counts"]["code"] == len(code_chunks)
    assert chunk_index["entries"][0]["text_preview"]
    assert chunk_index["entries"][0]["keywords"]
    assert retrieval_plan["paper_targets"]
    assert retrieval_plan["rebuttal_targets"]
    assert retrieval_plan["code_targets"]
    assert retrieval_plan["code_symbols"]
    assert retrieval_plan["code_symbols"][0]["path"]
    assert retrieval_plan["questions"]
    assert "method,kind,route_family" in result_ledger
    assert "baseline_fairness" in {item["concern_id"] for item in rebuttal_matrix["matrix"]}
    assert rebuttal_matrix["structured_concerns"][0]["source_snippet"]
    assert rebuttal_matrix["structured_concerns"][0]["experiment_implication"]
    assert rebuttal_matrix["structured_concerns"][0]["next_round_constraint"]
    assert "blocked_idea_patterns" in negative_memory


def test_tree_sitter_code_intake_builds_symbol_chunks_and_edges(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    aligner = repo / "rosetta/model/aligner.py"
    aligner.write_text(
        "\n".join(
            [
                "from rosetta.model.projector import Projector",
                "",
                "class CacheRouter:",
                "    def __init__(self, cfg):",
                "        self.cfg = cfg",
                "        self.projector = Projector()",
                "",
                "    def route(self, hidden, valid_mask):",
                "        gate = self.cfg.get('confidence_gate')",
                "        if valid_mask is None:",
                "            return hidden",
                "        return self.projector(hidden) * gate",
                "",
                "def build_router(cfg):",
                "    return CacheRouter(cfg)",
                "",
                "def run_route(router, hidden, valid_mask):",
                "    return router.route(hidden, valid_mask)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (repo / "test/test_cache_router.py").write_text(
        "from rosetta.model.aligner import CacheRouter\n\n"
        "def test_cache_router_route():\n"
        "    router = CacheRouter({'confidence_gate': 1.0})\n"
        "    assert router.route(1, True) == 1\n",
        encoding="utf-8",
    )
    (repo / "recipe/train_recipe/C2C_0.6+0.5.json").write_text(
        json.dumps({"alignment": {"confidence_gate": 1.0}, "output": {}, "data": {"kwargs": {}}}),
        encoding="utf-8",
    )
    paths = init_workspace(_base_config(tmp_path / "workspace"), "topic", project_id="proj_intake", simulate=True)
    config_patch = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "ref_paper": str(tmp_path / "missing_paper.txt"),
            "ref_rebuttal": str(tmp_path / "missing_rebuttal.txt"),
            "env_python": "/usr/bin/python3",
        }
    }
    adapter = C2CAdapter(paths.root, config_patch)

    intake = adapter.build_code_intake()

    symbols = {item["symbol"]: item for item in intake.symbols}
    assert "CacheRouter" in symbols
    assert "CacheRouter.route" in symbols
    assert "build_router" in symbols
    route_chunk = next(chunk for chunk in intake.chunks if chunk["symbol"] == "CacheRouter.route")
    assert route_chunk["node_type"] == "function_definition"
    assert route_chunk["edit_surface"] == "allowed"
    assert "valid_mask" in route_chunk["references"]
    assert "confidence_gate" in route_chunk["config_keys"]
    assert "alignment_core" in route_chunk["risk_tags"]
    assert any(edge["edge_type"] == "contains" and edge["dst"].endswith("CacheRouter.route") for edge in intake.edges)
    assert any(edge["edge_type"] == "calls" and edge["dst"] == "self.cfg.get" for edge in intake.edges)
    assert any(edge["edge_type"] == "resolved_call" and edge["call"] == "CacheRouter" for edge in intake.edges)
    assert any(edge["edge_type"] == "config_key_defined_in" and edge["config_key"] == "confidence_gate" for edge in intake.edges)
    assert any(edge["edge_type"] == "tests_symbol" for edge in intake.edges)
    assert intake.file_manifest["schema_version"] == "code_intake_v1"
    assert intake.file_manifest["parser_fingerprint"]["parser_config_hash"]
    assert intake.file_manifest["parser_fingerprint"]["chunking_config_hash"]
    assert intake.repo_map["counts"]["symbols"] == len(intake.symbols)
    assert intake.report["counts"]["chunks_with_config_keys"] >= 1
    assert intake.report["cache"]["counts"]["miss"] > 0
    assert intake.report["cache"]["parser_config_hash"] == intake.file_manifest["parser_fingerprint"]["parser_config_hash"]
    assert not intake.report["coverage"]["missing_allowed_files"]
    alignment_surface = intake.surface_map["surfaces"]["alignment_core"]
    assert any(item["symbol"] == "CacheRouter.route" for item in alignment_surface)
    retrieved = retrieve_code_chunks(query="confidence gate valid_mask routing", chunks=intake.chunks, top_k=3)
    assert retrieved[0]["symbol"] == "CacheRouter.route"
    assert any("confidence" in reason or "valid_mask" in reason for reason in retrieved[0]["match_reasons"])

    second_intake = adapter.build_code_intake()
    assert second_intake.report["cache"]["counts"]["hit"] >= intake.report["cache"]["counts"]["miss"]


class _FakeMinerUResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None, content: bytes = b""):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = content

    def json(self) -> dict:
        return self._payload


class _FakeMinerUSession:
    def __init__(self, zip_content: bytes):
        self.zip_content = zip_content
        self.requests: list[dict] = []

    def post(self, url: str, *, headers=None, json=None, timeout=None):
        self.requests.append({"method": "POST", "url": url, "headers": headers, "json": json, "timeout": timeout})
        return _FakeMinerUResponse(
            payload={
                "code": 0,
                "msg": "ok",
                "data": {"batch_id": "batch-1", "file_urls": ["https://upload.example/file.pdf"]},
            }
        )

    def put(self, url: str, *, data=None, timeout=None):
        self.requests.append({"method": "PUT", "url": url, "timeout": timeout})
        return _FakeMinerUResponse(status_code=200)

    def get(self, url: str, *, headers=None, timeout=None):
        self.requests.append({"method": "GET", "url": url, "headers": headers, "timeout": timeout})
        if "extract-results" in url:
            return _FakeMinerUResponse(
                payload={
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "batch_id": "batch-1",
                        "extract_result": [
                            {
                                "file_name": "paper.pdf",
                                "data_id": "paper-id",
                                "state": "done",
                                "full_zip_url": "https://download.example/result.zip",
                            }
                        ],
                    },
                }
            )
        return _FakeMinerUResponse(content=self.zip_content)


def _zip_with_full_md(markdown: str) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("nested/full.md", markdown)
    return buffer.getvalue()


def test_mineru_pdf_client_writes_paper_full_without_leaking_key(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    session = _FakeMinerUSession(_zip_with_full_md("# Title\n\n## Method\n\n$E=mc^2$\n"))
    client = MinerUPdfClient(
        api_key="secret-token",
        session=session,
        poll_interval_seconds=1,
        timeout_seconds=5,
    )

    result = client.parse_pdf(pdf_path, tmp_path / "out", data_id="paper-id", title="Fallback Title")

    paper_full = tmp_path / "out" / "paper_full.md"
    mineru_result = tmp_path / "out" / "mineru_result.json"
    assert result["state"] == "done"
    assert paper_full.exists()
    assert "## Method" in paper_full.read_text(encoding="utf-8")
    assert "secret-token" not in mineru_result.read_text(encoding="utf-8")
    assert session.requests[0]["headers"]["Authorization"] == "Bearer secret-token"


def test_mineru_pdf_client_requires_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MINERU_API_KEY", raising=False)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    client = MinerUPdfClient(api_key="", session=_FakeMinerUSession(_zip_with_full_md("# T\n")))

    try:
        client.parse_pdf(pdf_path, tmp_path / "out", data_id="paper-id")
    except MinerUError as exc:
        assert "MINERU_API_KEY" in str(exc)
    else:
        raise AssertionError("MinerUPdfClient should fail without an API key")


def test_mineru_pdf_client_wraps_connection_errors(tmp_path: Path) -> None:
    class BrokenMinerUSession:
        def post(self, *args, **kwargs):
            raise requests.ConnectionError("dns unavailable")

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    client = MinerUPdfClient(api_key="secret-token", session=BrokenMinerUSession())

    with pytest.raises(MinerUError, match="request upload URL"):
        client.parse_pdf(pdf_path, tmp_path / "out", data_id="paper-id")


def test_c2c_pdf_ref_uses_mineru_paper_full(monkeypatch, tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "paper.pdf"
    ref_rebuttal = tmp_path / "rebuttal.md"
    ref_paper.write_bytes(b"%PDF-1.4 fake")
    ref_rebuttal.write_text("review text", encoding="utf-8")
    paths = init_workspace(_base_config(tmp_path / "workspace"), "topic", project_id="proj_pdf", simulate=True)
    config_patch = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(source_repo),
            "ref_paper": str(ref_paper),
            "ref_rebuttal": str(ref_rebuttal),
            "env_python": "/usr/bin/python3",
            "pdf_ingest": {"provider": "mineru"},
        }
    }
    (paths.root / "meta/project_config.yaml").write_text(yaml.safe_dump(config_patch), encoding="utf-8")

    def fake_parse(self, pdf_path: Path, output_dir: Path, *, data_id: str, title: str = ""):
        del self, pdf_path, data_id, title
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "paper_full.md").write_text("# Paper\n\n## Method\n\n$$x+y$$\n", encoding="utf-8")
        result = {"provider": "mineru", "state": "done", "paper_full_md_path": "paper_full.md"}
        (output_dir / "mineru_result.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr("auto_research.c2c.MinerUPdfClient.parse_pdf", fake_parse)

    refs = C2CAdapter(paths.root, config_patch).import_reference_materials()

    assert refs["status"] == "ok"
    paper_card = next(card for card in refs["cards"] if card["kind"] == "ref_paper")
    assert paper_card["parser"] == "mineru"
    assert paper_card["paper_full_md_path"].endswith("paper_full.md")
    assert "## Method" in paper_card["text"]
    assert refs["paper_full_manifest"][0]["paper_full_md_path"] == paper_card["paper_full_md_path"]


def test_c2c_pdf_ref_reuses_mineru_sha_cache(monkeypatch, tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "paper.pdf"
    ref_rebuttal = tmp_path / "rebuttal.md"
    ref_paper.write_bytes(b"%PDF-1.4 fake cache")
    ref_rebuttal.write_text("review text", encoding="utf-8")
    paths = init_workspace(_base_config(tmp_path / "workspace"), "topic", project_id="proj_pdf_cache", simulate=True)
    config_patch = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(source_repo),
            "ref_paper": str(ref_paper),
            "ref_rebuttal": str(ref_rebuttal),
            "env_python": "/usr/bin/python3",
            "pdf_ingest": {"provider": "mineru"},
        }
    }
    calls = {"count": 0}

    def fake_parse(self, pdf_path: Path, output_dir: Path, *, data_id: str, title: str = ""):
        del self, pdf_path, data_id, title
        calls["count"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "paper_full.md").write_text("# Cached Paper\n\n## Method\n\ncache text\n", encoding="utf-8")
        result = {"provider": "mineru", "state": "done", "paper_full_md_path": "paper_full.md"}
        (output_dir / "mineru_result.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr("auto_research.c2c.MinerUPdfClient.parse_pdf", fake_parse)
    first = C2CAdapter(paths.root, config_patch).import_reference_materials()

    def fail_parse(self, pdf_path: Path, output_dir: Path, *, data_id: str, title: str = ""):
        del self, pdf_path, output_dir, data_id, title
        raise AssertionError("MinerU API should not be called when sha cache is available")

    monkeypatch.setattr("auto_research.c2c.MinerUPdfClient.parse_pdf", fail_parse)
    second = C2CAdapter(paths.root, config_patch).import_reference_materials()

    assert calls["count"] == 1
    assert first["paper_full_manifest"][0]["cache_status"] == "miss"
    assert first["paper_full_manifest"][0]["parser_config_hash"]
    assert first["paper_full_manifest"][0]["prompt_schema_version"] == "c2c_paper_full_markdown_v1"
    assert second["paper_full_manifest"][0]["cache_status"] in {"local_hit", "sha_hit"}
    assert second["paper_full_manifest"][0]["parser_config_hash"]
    assert "Cached Paper" in second["cards"][0]["text"]


def test_c2c_pdf_ref_reuses_shared_mineru_cache_across_projects(monkeypatch, tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "paper.pdf"
    ref_rebuttal = tmp_path / "rebuttal.md"
    ref_paper.write_bytes(b"%PDF-1.4 shared cache")
    ref_rebuttal.write_text("review text", encoding="utf-8")
    workspace = tmp_path / "workspace"
    first_paths = init_workspace(_base_config(workspace), "topic", project_id="proj_pdf_shared_a", simulate=True)
    second_paths = init_workspace(_base_config(workspace), "topic", project_id="proj_pdf_shared_b", simulate=True)
    config_patch = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(source_repo),
            "ref_paper": str(ref_paper),
            "ref_rebuttal": str(ref_rebuttal),
            "env_python": "/usr/bin/python3",
            "pdf_ingest": {"provider": "mineru"},
        }
    }
    calls = {"count": 0}

    def fake_parse(self, pdf_path: Path, output_dir: Path, *, data_id: str, title: str = ""):
        del self, pdf_path, data_id, title
        calls["count"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "paper_full.md").write_text("# Shared Paper\n\n## Method\n\nshared cache text\n", encoding="utf-8")
        result = {"provider": "mineru", "state": "done", "paper_full_md_path": "paper_full.md"}
        (output_dir / "mineru_result.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr("auto_research.c2c.MinerUPdfClient.parse_pdf", fake_parse)
    first = C2CAdapter(first_paths.root, config_patch).import_reference_materials()

    def fail_parse(self, pdf_path: Path, output_dir: Path, *, data_id: str, title: str = ""):
        del self, pdf_path, output_dir, data_id, title
        raise AssertionError("MinerU API should not be called when shared cache is available")

    monkeypatch.setattr("auto_research.c2c.MinerUPdfClient.parse_pdf", fail_parse)
    second = C2CAdapter(second_paths.root, config_patch).import_reference_materials()

    assert calls["count"] == 1
    assert first["paper_full_manifest"][0]["cache_status"] == "miss"
    assert second["paper_full_manifest"][0]["cache_status"] == "shared_hit"
    assert "Shared Paper" in second["cards"][0]["text"]


def test_c2c_pdf_ref_promotes_legacy_mineru_artifact_to_shared_cache(monkeypatch, tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "paper.pdf"
    ref_rebuttal = tmp_path / "rebuttal.md"
    ref_paper.write_bytes(b"%PDF-1.4 legacy cache")
    ref_rebuttal.write_text("review text", encoding="utf-8")
    workspace = tmp_path / "workspace"
    legacy_root = workspace / "legacy_project"
    legacy_md = legacy_root / "references/c2c/ref_paper/demo/paper_full.md"
    legacy_md.parent.mkdir(parents=True)
    legacy_md.write_text("# Legacy Paper\n\n## Method\n\nlegacy cache text\n", encoding="utf-8")
    write_json(
        legacy_root / "intake/c2c/static_bundle.json",
        {
            "schema_version": "c2c_static_intake_bundle_v1",
            "paper_full_manifest": [
                {
                    "paper_id": "demo",
                    "sha256": sha256_file(ref_paper),
                    "paper_full_md_path": "references/c2c/ref_paper/demo/paper_full.md",
                    "prompt_schema_version": "c2c_paper_full_markdown_v1",
                }
            ],
        },
    )
    paths = init_workspace(_base_config(workspace), "topic", project_id="proj_pdf_legacy", simulate=True)
    config_patch = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(source_repo),
            "ref_paper": str(ref_paper),
            "ref_rebuttal": str(ref_rebuttal),
            "env_python": "/usr/bin/python3",
            "pdf_ingest": {"provider": "mineru"},
        }
    }

    def fail_parse(self, pdf_path: Path, output_dir: Path, *, data_id: str, title: str = ""):
        del self, pdf_path, output_dir, data_id, title
        raise AssertionError("MinerU API should not be called when legacy artifact is available")

    monkeypatch.setattr("auto_research.c2c.MinerUPdfClient.parse_pdf", fail_parse)
    refs = C2CAdapter(paths.root, config_patch).import_reference_materials()

    assert refs["status"] == "ok"
    assert refs["paper_full_manifest"][0]["cache_status"] == "legacy_project_hit"
    assert "Legacy Paper" in refs["cards"][0]["text"]
    assert list((workspace / "_shared_cache" / "auto_research" / "mineru_pdf").glob("*/**/paper_full.md"))


def test_c2c_strong_reference_comparison_is_s3_only(tmp_path: Path) -> None:
    adapter = C2CAdapter(
        tmp_path,
        {
            "c2c": {
                "strong_references": [
                    {
                        "name": "strong_local_v22",
                        "mean": 50.82,
                        "datasets": {"mmlu-redux": 47.07, "ai2-arc": 54.78, "openbookqa": 50.60},
                        "visible_to_ideation": False,
                        "reference_role": "s3_strong_reference_only",
                    },
                    {
                        "name": "ideation_visible_ref",
                        "mean": 99.0,
                        "datasets": {"mmlu-redux": 99.0},
                        "visible_to_ideation": True,
                    },
                ]
            }
        },
    )
    best = {
        "id": "winner",
        "metrics": {"mean": 51.0, "datasets": {"mmlu-redux": 48.0, "ai2-arc": 55.0, "openbookqa": 50.0}},
    }

    comparisons = ExperimentAgent._c2c_strong_reference_comparisons(best, adapter)

    assert [item["name"] for item in comparisons] == ["strong_local_v22"]
    assert comparisons[0]["delta_vs_reference"] == 0.18
    assert comparisons[0]["used_for_acceptance"] is False
    assert comparisons[0]["visible_to_ideation"] is False


def test_c2c_patch_guard_rejects_out_of_scope_and_applies_allowed(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    guard = C2CPatchGuard(["rosetta/model/aligner.py"], ["recipe/", "local/auto_research_runs/"])

    rejected = guard.apply_edits(repo, [{"path": "script/train/SFT_train.py", "old": "train", "new": "patch"}])
    applied = guard.apply_edits(repo, [{"path": "rosetta/model/aligner.py", "old": "aligner", "new": "patched"}])

    assert rejected["status"] == "rejected"
    assert applied["status"] == "applied"
    assert "patched" in (repo / "rosetta/model/aligner.py").read_text(encoding="utf-8")


def test_dynamic_edit_policy_allows_expected_scope_and_rejects_forbidden_paths(tmp_path: Path) -> None:
    policy = DynamicEditPolicy.from_config()

    allowed = [
        "rosetta/model/aligner.py",
        "script/train/SFT_train.py",
        "recipe/train_recipe/demo.json",
        "recipe/eval_recipe/demo.yaml",
        "test/test_aligner_span_overlap.py",
        "tests/test_patch.py",
        "pyproject.toml",
        "requirements-dev.txt",
    ]
    rejected = [
        "/tmp/outside.py",
        "../outside.py",
        "local/final_results/old.py",
        "local/checkpoints/model.py",
        "data/cache.py",
        "datasets/mmlu.py",
        "models/model.py",
        "rosetta/model/weights.bin",
        "foo/requirements-dev.txt",
    ]

    assert all(policy.allowed(path) for path in allowed)
    assert not any(policy.allowed(path) for path in rejected)

    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "escape.py").write_text("VALUE = 1\n", encoding="utf-8")
    repo.mkdir()
    (repo / "rosetta").symlink_to(outside, target_is_directory=True)

    assert not policy.allowed("rosetta/escape.py", repo_root=repo)


def test_frozen_patch_guard_requires_sha_and_restores_added_files(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    policy = DynamicEditPolicy.from_config()
    guard = FrozenPatchGuard(policy)
    aligner = repo / "rosetta/model/aligner.py"
    original = aligner.read_text(encoding="utf-8")
    added = repo / "test/test_added_patch.py"

    result = guard.apply(
        repo,
        {
            "operations": [
                {
                    "op": "replace_file",
                    "path": "rosetta/model/aligner.py",
                    "old_sha256": sha256_file(aligner),
                    "new": "VALUE = 'patched'\n",
                },
                {
                    "op": "add_file",
                    "path": "test/test_added_patch.py",
                    "new": "def test_added_patch():\n    assert True\n",
                },
            ]
        },
    )

    assert result["status"] == "applied"
    assert "patched" in aligner.read_text(encoding="utf-8")
    assert added.exists()

    guard.restore(repo, result["restore_state"])

    assert aligner.read_text(encoding="utf-8") == original
    assert not added.exists()

    missing_sha = guard.apply(
        repo,
        {
            "operations": [
                {"op": "replace_file", "path": "rosetta/model/aligner.py", "new": "VALUE = 'bad'\n"}
            ]
        },
    )
    bad_sha = guard.apply(
        repo,
        {
            "operations": [
                {
                    "op": "replace_file",
                    "path": "rosetta/model/aligner.py",
                    "old_sha256": "not-the-current-sha",
                    "new": "VALUE = 'bad'\n",
                }
            ]
        },
    )
    forbidden = guard.apply(
        repo,
        {
            "operations": [
                {"op": "add_file", "path": "local/final_results/old.py", "new": "VALUE = 'bad'\n"}
            ]
        },
    )

    assert missing_sha["status"] == "rejected"
    assert bad_sha["status"] == "rejected"
    assert forbidden["status"] == "rejected"
    assert aligner.read_text(encoding="utf-8") == original


def _code_patch_test_config(workspace_root: Path, repo: Path, *, require_targeted_tests: bool = False) -> dict:
    config = _base_config(workspace_root, simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {"enabled": False},
        },
    }
    config["code_patch"] = {
        "enabled": True,
        "backend": "mock_codex",
        "worktree_storage_root": str(workspace_root.parent / "code_worktrees_cache"),
        "timeout_seconds": 1800,
        "max_candidates": 3,
        "variants_per_candidate": 1,
        "validation": {
            "require_py_compile": True,
            "require_targeted_tests": require_targeted_tests,
            "runtime_smoke": {"enabled": False},
            "mechanism_self_review": {"enabled": False},
        },
    }
    return config


def test_code_patch_agent_generates_artifacts_from_temp_repo_without_polluting_source(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class MockBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            assert implementation_contract["candidate_id"] == "idea_patch"
            assert "implementation_targets" in implementation_contract
            assert edit_policy.allowed("script/train/SFT_train.py", repo_root=temp_repo)
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'patched aligner'\n", encoding="utf-8")
            (temp_repo / "script/train/SFT_train.py").write_text("print('patched train')\n", encoding="utf-8")
            (temp_repo / ".pytest_cache/v/cache").mkdir(parents=True)
            (temp_repo / ".pytest_cache/v/cache/nodeids").write_text("[]\n", encoding="utf-8")
            (temp_repo / "rosetta/model/__pycache__").mkdir(parents=True)
            (temp_repo / "rosetta/model/__pycache__/aligner.cpython-310.pyc").write_bytes(b"cache")
            (temp_repo / ".coverage").write_text("coverage-data\n", encoding="utf-8")
            (temp_repo / "htmlcov").mkdir()
            (temp_repo / "htmlcov/index.html").write_text("<html></html>\n", encoding="utf-8")
            (temp_repo / "test/test_patch_backend.py").write_text(
                "def test_patch_backend():\n    assert True\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Implemented a multi-file patch in the temporary repo."}

    ideas = [{"id": "idea_patch", "title": "Patch Idea", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=MockBackend()).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    assert manifest["selected_candidate_id"] == "idea_patch"
    assert manifest["valid_patch_ids"] == ["idea_patch"]
    assert manifest["selected_patch"]["candidate_id"] == "idea_patch"
    assert manifest["selected_patch"]["patch_json"].endswith("plan/code_patches/idea_patch/patch.json")
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert ideas[0]["code_patch"]["has_executable_change"] is True
    assert set(ideas[0]["code_patch"]["changed_files"]) == {
        "rosetta/model/aligner.py",
        "script/train/SFT_train.py",
        "test/test_patch_backend.py",
    }
    assert (paths.root / "plan/code_patches/idea_patch/patch.json").exists()
    assert (paths.root / "plan/code_patches/idea_patch/patch.diff").exists()
    assert (paths.root / "plan/code_patches/idea_patch/rationale.md").exists()
    assert (paths.root / "plan/code_patches/idea_patch/validation.json").exists()
    assert (paths.root / "plan/code_patches/idea_patch/implementation_contract.json").exists()
    assert (paths.root / "plan/code_patches/idea_patch/codex_prompt.md").exists()
    assert (paths.root / "plan/code_patches/patch_manifest.json").exists()
    prompt = (paths.root / "plan/code_patches/idea_patch/codex_prompt.md").read_text(encoding="utf-8")
    assert "Implementation contract" in prompt
    assert "aligner" in (repo / "rosetta/model/aligner.py").read_text(encoding="utf-8")
    assert "patched" not in (repo / "rosetta/model/aligner.py").read_text(encoding="utf-8")
    assert "patched" not in (repo / "script/train/SFT_train.py").read_text(encoding="utf-8")


def test_code_patch_delta_ignores_c2c_generated_runtime_artifacts(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["validation"]["runtime_smoke"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_patch_runtime_artifacts", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class RuntimeArtifactBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'patched aligner'\n", encoding="utf-8")
            baseline_final = temp_repo / "local/auto_research_runs/proxy_baseline/checkpoints/final"
            baseline_final.mkdir(parents=True, exist_ok=True)
            (baseline_final / "projector_0.json").write_text('{"generated": true}\n', encoding="utf-8")
            baseline_results = temp_repo / "local/auto_research_runs/proxy_baseline/results/mmlu-redux"
            baseline_results.mkdir(parents=True, exist_ok=True)
            (baseline_results / "Rosetta_mmlu-redux_generate_summary.json").write_text(
                '{"overall_accuracy": 0.5}\n',
                encoding="utf-8",
            )
            candidate_root = temp_repo / "local/auto_research_runs/idea_patch"
            candidate_root.mkdir(parents=True, exist_ok=True)
            (candidate_root / "train_recipe.json").write_text('{"generated": true}\n', encoding="utf-8")
            (candidate_root / "eval_mmlu-redux.yaml").write_text("generated: true\n", encoding="utf-8")
            (candidate_root / "run_state.json").write_text('{"generated": true}\n', encoding="utf-8")
            candidate_final = temp_repo / "local/auto_research_runs/idea_patch/checkpoints/final"
            candidate_final.mkdir(parents=True, exist_ok=True)
            (candidate_final / "adapter.bin").write_bytes(b"generated")
            return {"status": "ok", "rationale": "Patched model code and generated runtime artifacts."}

    ideas = [{"id": "idea_patch", "title": "Patch Idea", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=RuntimeArtifactBackend()).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["changed_files"] == ["rosetta/model/aligner.py"]
    patch = json.loads((paths.root / "plan/code_patches/idea_patch/patch.json").read_text(encoding="utf-8"))
    assert [operation["path"] for operation in patch["operations"]] == ["rosetta/model/aligner.py"]
    diff = (paths.root / "plan/code_patches/idea_patch/patch.diff").read_text(encoding="utf-8")
    assert "local/auto_research_runs" not in diff


def test_code_patch_persistent_backend_uses_git_worktree_and_codex_resume(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_git_c2c_repo(tmp_path)
    snapshot = tmp_path / "snapshot"
    import shutil

    shutil.copytree(repo, snapshot, ignore=lambda directory, names: {".git"} & set(names))
    config = _code_patch_test_config(tmp_path / "workspace", snapshot)
    config["c2c"]["target_repo"] = str(repo)
    config["code_patch"]["backend"] = "codex_persistent_cli"
    config["code_patch"]["persistent_session"] = True
    config["code_patch"]["use_git_worktree"] = True
    config["code_patch"]["codex_json_events"] = True
    config["code_patch"]["validation"]["require_py_compile"] = False
    config["code_patch"]["validation"]["runtime_smoke"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_persistent", simulate=False)
    artifacts = ArtifactManager(paths.root)
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append([str(part) for part in command])
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["worktree", "add"]:
            worktree_repo = Path(command[-2])
            shutil.copytree(snapshot, worktree_repo)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command and command[0] == "codex":
            output_path = Path(command[command.index("--output-last-message") + 1])
            if "resume" in command:
                (Path(kwargs["cwd"]) / "rosetta/model/aligner.py").write_text("VALUE = 'persistent'\n", encoding="utf-8")
                output_path.write_text("patched\n", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout='{"type":"event","stage":"patch"}\n', stderr="session id: 123e4567-e89b-12d3-a456-426614174000\n")
            output_path.write_text('{"files":["rosetta/model/aligner.py"]}\n', encoding="utf-8")
            return SimpleNamespace(
                returncode=0,
                stdout='{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614174000"}\n{"type":"event","stage":"preload"}\n',
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    import auto_research.code_patch as code_patch_module

    monkeypatch.setattr(code_patch_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(code_patch_module.subprocess, "run", fake_run)

    ideas = [{"id": "idea/session path", "title": "Persistent", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    session_dir = paths.root / "plan/code_worktrees/idea-session_path/v1"
    metadata = json.loads((session_dir / "worktree_metadata.json").read_text(encoding="utf-8"))
    repo_path = Path(metadata["repo"])
    assert repo_path.exists()
    assert repo_path.is_relative_to(Path(config["code_patch"]["worktree_storage_root"]))
    assert not (session_dir / "repo").exists()
    assert (session_dir / "codex_session.json").exists()
    assert (session_dir / "codex_events.jsonl").exists()
    assert (session_dir / "patch_blueprint.json").exists()
    assert "123e4567-e89b-12d3-a456-426614174000" in (paths.root / "meta/codex_sessions.yaml").read_text(encoding="utf-8")
    codex_commands = [command for command in commands if command and command[0] == "codex"]
    assert len(codex_commands) == 2
    assert "resume" not in codex_commands[0]
    assert "resume" in codex_commands[1]
    assert "--json" in codex_commands[0]
    assert "--json" in codex_commands[1]
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))
    assert patch["code_worktree"]["branch"] == "auto-research/proj_persistent/idea-session_path/v1"
    assert ideas[0]["code_patch"]["codex_session_id"] == "123e4567-e89b-12d3-a456-426614174000"


def test_code_patch_persistent_backend_materializes_snapshot_over_git_head(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_git_c2c_repo(tmp_path)
    snapshot = tmp_path / "snapshot"
    import shutil

    shutil.copytree(repo, snapshot, ignore=lambda directory, names: {".git"} & set(names))
    baseline_text = (snapshot / "rosetta/model/aligner.py").read_text(encoding="utf-8")
    config = _code_patch_test_config(tmp_path / "workspace", snapshot)
    config["c2c"]["target_repo"] = str(repo)
    config["code_patch"]["backend"] = "codex_persistent_cli"
    config["code_patch"]["persistent_session"] = True
    config["code_patch"]["use_git_worktree"] = True
    config["code_patch"]["validation"]["require_py_compile"] = False
    config["code_patch"]["validation"]["runtime_smoke"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_materialize_snapshot", simulate=False)
    artifacts = ArtifactManager(paths.root)

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="stalehead\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["worktree", "add"]:
            worktree_repo = Path(command[-2])
            shutil.copytree(snapshot, worktree_repo)
            (worktree_repo / "rosetta/model/aligner.py").write_text("VALUE = 'stale-head'\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command and command[0] == "codex":
            worktree_repo = Path(kwargs["cwd"])
            assert (worktree_repo / "rosetta/model/aligner.py").read_text(encoding="utf-8") == baseline_text
            output_path = Path(command[command.index("--output-last-message") + 1])
            if "resume" in command:
                (worktree_repo / "rosetta/model/aligner.py").write_text(baseline_text + "PATCHED = True\n", encoding="utf-8")
                output_path.write_text("patched\n", encoding="utf-8")
            else:
                output_path.write_text('{"files":["rosetta/model/aligner.py"]}\n', encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="session id: 123e4567-e89b-12d3-a456-426614174000\n")
        raise AssertionError(f"unexpected command: {command}")

    import auto_research.code_patch as code_patch_module

    monkeypatch.setattr(code_patch_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(code_patch_module.subprocess, "run", fake_run)

    ideas = [{"id": "materialize", "title": "Materialize", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    metadata = json.loads((paths.root / "plan/code_worktrees/materialize/v1/worktree_metadata.json").read_text(encoding="utf-8"))
    assert metadata["baseline_materialized_from_snapshot"] is True
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))
    operation = patch["operations"][0]
    assert operation["path"] == "rosetta/model/aligner.py"
    assert operation["old_sha256"] == sha256_file(snapshot / "rosetta/model/aligner.py")


def test_code_patch_persistent_backend_reuses_existing_worktree(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_git_c2c_repo(tmp_path)
    snapshot = tmp_path / "snapshot"
    import shutil

    shutil.copytree(repo, snapshot, ignore=lambda directory, names: {".git"} & set(names))
    config = _code_patch_test_config(tmp_path / "workspace", snapshot)
    config["c2c"]["target_repo"] = str(repo)
    config["code_patch"]["backend"] = "codex_persistent_cli"
    config["code_patch"]["persistent_session"] = True
    config["code_patch"]["use_git_worktree"] = True
    config["code_patch"]["validation"]["require_py_compile"] = False
    config["code_patch"]["validation"]["runtime_smoke"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_reuse_worktree", simulate=False)
    artifacts = ArtifactManager(paths.root)
    existing_repo = paths.root / "plan/code_worktrees/reuse/v1/repo"
    shutil.copytree(snapshot, existing_repo)
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append([str(part) for part in command])
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["worktree", "add"]:
            raise AssertionError("worktree add should not run for existing worktree")
        if command and command[0] == "codex":
            output_path = Path(command[command.index("--output-last-message") + 1])
            if "resume" in command:
                (Path(kwargs["cwd"]) / "rosetta/model/aligner.py").write_text("VALUE = 'reuse'\n", encoding="utf-8")
            output_path.write_text("ok\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="session id: 123e4567-e89b-12d3-a456-426614174000\n")
        raise AssertionError(f"unexpected command: {command}")

    import auto_research.code_patch as code_patch_module

    monkeypatch.setattr(code_patch_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(code_patch_module.subprocess, "run", fake_run)

    ideas = [{"id": "reuse", "title": "Reuse", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    assert not any(command[:5] == ["git", "-C", str(repo), "worktree", "add"] for command in commands)


def test_code_patch_persistent_backend_resume_failure_falls_back_to_new_session(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_git_c2c_repo(tmp_path)
    snapshot = tmp_path / "snapshot"
    import shutil

    shutil.copytree(repo, snapshot, ignore=lambda directory, names: {".git"} & set(names))
    config = _code_patch_test_config(tmp_path / "workspace", snapshot)
    config["c2c"]["target_repo"] = str(repo)
    config["code_patch"]["backend"] = "codex_persistent_cli"
    config["code_patch"]["persistent_session"] = True
    config["code_patch"]["use_git_worktree"] = True
    config["code_patch"]["validation"]["require_py_compile"] = False
    config["code_patch"]["validation"]["runtime_smoke"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_resume_fallback", simulate=False)
    artifacts = ArtifactManager(paths.root)
    session_dir = paths.root / "plan/code_worktrees/fallback/v1"
    session_dir.mkdir(parents=True)
    (session_dir / "codex_session.json").write_text(json.dumps({"session_id": "old-session"}), encoding="utf-8")
    resume_attempts = 0

    def fake_run(command, **kwargs):
        nonlocal resume_attempts
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["worktree", "add"]:
            shutil.copytree(snapshot, Path(command[-2]))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command and command[0] == "codex":
            output_path = Path(command[command.index("--output-last-message") + 1])
            if "resume" in command:
                resume_attempts += 1
                output_path.write_text("", encoding="utf-8")
                return SimpleNamespace(returncode=1, stdout="", stderr="session not found\n")
            (Path(kwargs["cwd"]) / "rosetta/model/aligner.py").write_text("VALUE = 'fallback'\n", encoding="utf-8")
            output_path.write_text("patched\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="session id: 123e4567-e89b-12d3-a456-426614174000\n")
        raise AssertionError(f"unexpected command: {command}")

    import auto_research.code_patch as code_patch_module

    monkeypatch.setattr(code_patch_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(code_patch_module.subprocess, "run", fake_run)

    ideas = [{"id": "fallback", "title": "Fallback", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    assert resume_attempts == 1
    actions = ideas[0]["code_patch"].get("recovery_actions") or []
    assert any(action.get("action") == "retry_codex_with_new_persistent_session" for action in actions)
    assert "123e4567-e89b-12d3-a456-426614174000" in (session_dir / "codex_session.json").read_text(encoding="utf-8")


def test_code_patch_implementation_failure_reuses_patch_session(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_git_c2c_repo(tmp_path)
    snapshot = tmp_path / "snapshot"
    import shutil

    shutil.copytree(repo, snapshot, ignore=lambda directory, names: {".git"} & set(names))
    config = _code_patch_test_config(tmp_path / "workspace", snapshot)
    config["c2c"]["target_repo"] = str(repo)
    config["code_patch"]["backend"] = "codex_persistent_cli"
    config["code_patch"]["persistent_session"] = True
    config["code_patch"]["use_git_worktree"] = True
    config["code_patch"]["validation"]["require_py_compile"] = False
    config["code_patch"]["validation"]["runtime_smoke"] = {"enabled": False}
    config["c2c"]["env_python"] = "/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python"
    paths = init_workspace(config, "topic", project_id="proj_fresh_patch_session", simulate=False)
    artifacts = ArtifactManager(paths.root)
    write_json(
        paths.root / "plan" / "code_patches" / "fresh" / "validation.json",
        {
            "status": "validation_failed",
            "checks": [
                {
                    "name": "runtime_smoke:mechanism_activation_forward_probe",
                    "returncode": 1,
                    "failure_category": "mechanism_activation_forward_probe_failed",
                    "probe": {
                        "tensor_checks": [
                            {
                                "name": "projector_output",
                                "changed": False,
                                "enabled_sha256": "aaa",
                                "disabled_sha256": "aaa",
                            }
                        ],
                        "switch_seen_by_forward": False,
                        "cache_key_diff": 0.0,
                        "cache_value_diff": 0.0,
                        "projector_called": True,
                        "projector_output_identical": True,
                    },
                }
            ],
        },
    )
    write_json(
        paths.root / "plan" / "code_patches" / "patch_manifest.json",
        {
            "status": "no_valid_patch",
            "candidates": [
                {
                    "candidate_id": "fresh",
                    "status": "validation_failed",
                    "changed_files": ["rosetta/model/projector.py"],
                    "validation": "plan/code_patches/fresh/validation.json",
                }
            ],
        },
    )
    session_dir = paths.root / "plan/code_worktrees/fresh/v1"
    existing_repo = session_dir / "repo"
    session_dir.mkdir(parents=True)
    shutil.copytree(snapshot, existing_repo)
    (session_dir / "codex_session.json").write_text(json.dumps({"session_id": "old-session"}), encoding="utf-8")
    codex_commands: list[list[str]] = []
    codex_prompts: list[str] = []

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["worktree", "add"]:
            raise AssertionError("existing worktree should be reused")
        if command and command[0] == "codex":
            command_text = [str(part) for part in command]
            prompt = str(kwargs.get("input") or "")
            codex_commands.append(command_text)
            codex_prompts.append(prompt)
            assert "resume" in command_text
            assert "old-session" in command_text
            output_path = Path(command[command.index("--output-last-message") + 1])
            if "root-cause diagnosis pre-pass" in prompt:
                assert "/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python" in prompt
                assert "Do not edit files in this turn" in prompt
                assert "Do not run full train" in prompt
                assert "repeated_failure_context" in prompt
                assert "same_switch_seen_by_forward_false" in prompt
                assert "same_identical_tensors:projector_output" in prompt
                output_path.write_text(
                    json.dumps(
                        {
                            "root_cause": "ablation switch reaches config but projector.forward never reads it",
                            "evidence": ["config_overrides has disable_x=true"],
                            "repair_target": ["rosetta/model/projector.py"],
                            "forbidden": ["do not change evaluator"],
                            "lightweight_commands_run": [
                                "/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m py_compile rosetta/model/projector.py"
                            ],
                            "env_python_used": "/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python",
                            "confidence": "high",
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            assert "repair_diagnosis" in prompt
            assert "repeated_failure_context" in prompt
            assert "ordinary same-path repair has already failed" in prompt
            (Path(kwargs["cwd"]) / "rosetta/model/aligner.py").write_text("VALUE = 'reused patch session'\n", encoding="utf-8")
            output_path.write_text("patched\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    import auto_research.code_patch as code_patch_module

    monkeypatch.setattr(code_patch_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(code_patch_module.subprocess, "run", fake_run)

    ideas = [
        {
            "id": "fresh",
            "title": "Fresh",
            "hypothesis": "h",
            "previous_patch_failure": {
                "failure_class": "implementation_failure",
                "s2_5_repair_dispatch": {
                    "mode": "s2_5_only_implementation_repair",
                    "selected_candidate_id": "fresh",
                    "changed_files": ["rosetta/model/projector.py"],
                    "activation_forward_probe_diagnostics": {
                        "identical_tensors": ["projector_output"],
                        "changed_tensors": [],
                        "switch_seen_by_forward": False,
                        "cache_key_diff": 0.0,
                        "cache_value_diff": 0.0,
                        "projector_output_identical": True,
                    },
                    "tensor_checks": {"identical_tensors": ["projector_output"], "changed_tensors": []},
                },
                "proxy_effect_repair_contract": {
                    "mode": "s2_5_only_implementation_repair",
                },
            },
        }
    ]
    manifest = CodePatchAgent(paths.root, config, artifacts).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    assert len(codex_commands) == 2
    assert "resume" in codex_commands[0]
    assert "old-session" in codex_commands[0]
    assert "resume" in codex_commands[1]
    assert "old-session" in codex_commands[1]
    assert "root-cause diagnosis pre-pass" in codex_prompts[0]
    assert "repair_diagnosis" in codex_prompts[1]
    actions = ideas[0]["code_patch"].get("recovery_actions") or []
    assert any(action.get("action") == "s2_5_implementation_repair_diagnosis" for action in actions)
    assert not any(action.get("action") == "discard_patch_codex_session_for_implementation_repair" for action in actions)
    metadata = json.loads((session_dir / "worktree_metadata.json").read_text(encoding="utf-8"))
    assert metadata["session_policy"] == "persistent_resume_required"
    assert ideas[0]["code_patch"]["codex_session_id"] == "old-session"
    diagnosis = json.loads((paths.root / "plan" / "code_patches" / "fresh" / "repair_diagnosis.json").read_text(encoding="utf-8"))
    assert diagnosis["root_cause"] == "ablation switch reaches config but projector.forward never reads it"
    assert diagnosis["env_python_used"] == "/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python"
    assert diagnosis["same_session_reused"] is True
    repeated = diagnosis["implementation_repair_diagnosis"]["repeated_failure_context"]
    assert repeated["is_repeated"] is True
    assert "same_switch_seen_by_forward_false" in repeated["repeated_signals"]
    assert "same_identical_tensors:projector_output" in repeated["repeated_signals"]
    patch = json.loads((paths.root / "plan" / "code_patches" / "fresh" / "patch.json").read_text(encoding="utf-8"))
    assert patch["repair_diagnosis"]["root_cause"] == diagnosis["root_cause"]
    assert patch["implementation_contract"]["repeated_failure_context"]["is_repeated"] is True


def test_code_patch_persistent_validation_repair_uses_same_codex_session(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_git_c2c_repo(tmp_path)
    snapshot = tmp_path / "snapshot"
    import shutil

    shutil.copytree(repo, snapshot, ignore=lambda directory, names: {".git"} & set(names))
    config = _code_patch_test_config(tmp_path / "workspace", snapshot, require_targeted_tests=True)
    config["c2c"]["target_repo"] = str(repo)
    config["code_patch"]["backend"] = "codex_persistent_cli"
    config["code_patch"]["persistent_session"] = True
    config["code_patch"]["use_git_worktree"] = True
    config["code_patch"]["codex_json_events"] = True
    config["code_patch"]["validation"]["require_py_compile"] = False
    config["code_patch"]["validation"]["runtime_smoke"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_persistent_repair_session", simulate=False)
    artifacts = ArtifactManager(paths.root)
    codex_commands: list[list[str]] = []
    codex_prompts: list[str] = []

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["worktree", "add"]:
            shutil.copytree(snapshot, Path(command[-2]))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command and command[0] == "codex":
            command_text = [str(part) for part in command]
            prompt = str(kwargs.get("input") or "")
            codex_commands.append(command_text)
            codex_prompts.append(prompt)
            output_path = Path(command[command.index("--output-last-message") + 1])
            worktree_repo = Path(kwargs["cwd"])
            if "Persistent S2.5 Codex session bootstrap" in prompt:
                output_path.write_text("blueprint\n", encoding="utf-8")
                return SimpleNamespace(
                    returncode=0,
                    stdout='{"type":"thread.started","thread_id":"repair-session"}\n',
                    stderr="",
                )
            if "codex_repair_packet" in prompt:
                (worktree_repo / "test/test_aligner_span_overlap.py").write_text(
                    "def test_span():\n    assert True\n",
                    encoding="utf-8",
                )
                output_path.write_text("repaired\n", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="session id: repair-session\n")
            (worktree_repo / "rosetta/model/aligner.py").write_text("VALUE = 'persistent validation repair'\n", encoding="utf-8")
            (worktree_repo / "test/test_aligner_span_overlap.py").write_text(
                "def test_span():\n    assert False\n",
                encoding="utf-8",
            )
            output_path.write_text("initial patch\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="session id: repair-session\n")
        if "-m" in command and "pytest" in command:
            test_path = Path(kwargs["cwd"]) / command[-1]
            if "assert False" in test_path.read_text(encoding="utf-8"):
                return SimpleNamespace(returncode=1, stdout="FAILED test_span\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="1 passed\n", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    import auto_research.code_patch as code_patch_module

    monkeypatch.setattr(code_patch_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(code_patch_module.subprocess, "run", fake_run)

    ideas = [{"id": "persistent_repair", "title": "Persistent Repair", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts).run({"candidate_ideas": ideas}, ideas)
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert len(codex_commands) == 3
    assert "resume" not in codex_commands[0]
    assert codex_commands[1][-3:] == ["resume", "repair-session", "-"]
    assert codex_commands[2][-3:] == ["resume", "repair-session", "-"]
    assert "codex_repair_packet" in codex_prompts[2]
    action = next(
        action
        for action in patch["recovery_actions"]
        if action.get("action") == "retry_codex_after_validation_failure"
    )
    assert action["action"] == "retry_codex_after_validation_failure"
    assert action["repair_session"]["same_session_reused"] is True
    assert action["repair_session"]["session_id_before"] == "repair-session"
    assert action["repair_session"]["session_id_after"] == "repair-session"


def test_code_patch_prunes_persistent_worktree_before_codex(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_git_c2c_repo(tmp_path)
    snapshot = tmp_path / "snapshot"
    import shutil

    shutil.copytree(repo, snapshot, ignore=lambda directory, names: {".git"} & set(names))
    (snapshot / "rosetta/train").mkdir(parents=True, exist_ok=True)
    (snapshot / "rosetta/train/model_utils.py").write_text("VALUE = 'model utils baseline'\n", encoding="utf-8")
    config = _code_patch_test_config(tmp_path / "workspace", snapshot)
    config["c2c"]["target_repo"] = str(repo)
    config["code_patch"]["backend"] = "codex_persistent_cli"
    config["code_patch"]["persistent_session"] = True
    config["code_patch"]["use_git_worktree"] = True
    config["code_patch"]["validation"]["require_py_compile"] = False
    config["code_patch"]["validation"]["runtime_smoke"] = {"enabled": False}
    config["code_patch"]["validation"]["max_changed_files"] = 4
    paths = init_workspace(config, "topic", project_id="proj_prune_before_codex", simulate=False)
    artifacts = ArtifactManager(paths.root)
    session_dir = paths.root / "plan/code_worktrees/prune_before/v1"
    existing_repo = session_dir / "repo"
    session_dir.mkdir(parents=True)
    shutil.copytree(snapshot, existing_repo)
    (session_dir / "codex_session.json").write_text(json.dumps({"session_id": "stale-session"}), encoding="utf-8")
    (session_dir / "worktree_metadata.json").write_text(
        json.dumps(
            {
                "baseline_materialized_from_snapshot": True,
                "baseline_guard": {"snapshot_root": str(snapshot.resolve())},
            }
        ),
        encoding="utf-8",
    )
    (existing_repo / "script/evaluation/unified_evaluator.py").write_text("print('stale eval edit')\n", encoding="utf-8")
    (existing_repo / "script/train/SFT_train.py").write_text("print('stale train edit')\n", encoding="utf-8")
    (existing_repo / "rosetta/train/model_utils.py").write_text("VALUE = 'stale helper edit'\n", encoding="utf-8")
    baseline_eval = (snapshot / "script/evaluation/unified_evaluator.py").read_text(encoding="utf-8")
    baseline_train = (snapshot / "script/train/SFT_train.py").read_text(encoding="utf-8")
    baseline_helper = (snapshot / "rosetta/train/model_utils.py").read_text(encoding="utf-8")
    codex_calls = 0

    def fake_run(command, **kwargs):
        nonlocal codex_calls
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["worktree", "add"]:
            raise AssertionError("existing worktree should be reused")
        if command and command[0] == "codex":
            worktree_repo = Path(kwargs["cwd"])
            output_path = Path(command[command.index("--output-last-message") + 1])
            prompt = str(kwargs.get("input") or "")
            if "Persistent S2.5 Codex session bootstrap" in prompt:
                output_path.write_text("preload\n", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="session id: preload-session\n")
            if "Persistent S2.5 Codex root-cause diagnosis" in prompt:
                output_path.write_text(
                    json.dumps({"root_cause": "stale-worktree", "evidence": "test", "repair_target": "aligner.py"}),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="session id: stale-session\n")
            codex_calls += 1
            assert (worktree_repo / "script/evaluation/unified_evaluator.py").read_text(encoding="utf-8") == baseline_eval
            assert (worktree_repo / "script/train/SFT_train.py").read_text(encoding="utf-8") == baseline_train
            assert (worktree_repo / "rosetta/train/model_utils.py").read_text(encoding="utf-8") == baseline_helper
            (worktree_repo / "rosetta/model/aligner.py").write_text("VALUE = 'focused patch'\n", encoding="utf-8")
            output_path.write_text("patched\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="session id: focused-session\n")
        raise AssertionError(f"unexpected command: {command}")

    import auto_research.code_patch as code_patch_module

    monkeypatch.setattr(code_patch_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(code_patch_module.subprocess, "run", fake_run)

    ideas = [
        {
            "id": "prune_before",
            "title": "Prune Before",
            "hypothesis": "h",
            "experiment_contract": {
                "expected_files": ["rosetta/model/aligner.py", "test/test_aligner_span_overlap.py"]
            },
            "previous_patch_failure": {
                "failure_class": "implementation_failure",
                "proxy_effect_repair_contract": {
                    "mode": "implementation_patch_repair",
                },
            },
        }
    ]
    manifest = CodePatchAgent(paths.root, config, artifacts).run({"candidate_ideas": ideas}, ideas)
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))

    assert codex_calls == 1
    assert manifest["status"] == "ok"
    assert patch["changed_files"] == ["rosetta/model/aligner.py"]
    actions = patch.get("recovery_actions") or []
    prune_action = next(action for action in actions if action.get("action") == "auto_prune_worktree_scope_before_codex")
    assert set(prune_action["restored_files"]) == {
        "script/evaluation/unified_evaluator.py",
        "script/train/SFT_train.py",
        "rosetta/train/model_utils.py",
    }


def test_code_patch_persistent_backend_rejects_non_git_target_repo(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["c2c"]["target_repo"] = str(repo)
    config["code_patch"]["backend"] = "codex_persistent_cli"
    config["code_patch"]["persistent_session"] = True
    config["code_patch"]["use_git_worktree"] = True
    paths = init_workspace(config, "topic", project_id="proj_non_git", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [{"id": "non_git", "title": "Non Git", "hypothesis": "h"}]

    manifest = CodePatchAgent(paths.root, config, artifacts).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "no_valid_patch"
    assert "Git worktree requires c2c.target_repo to be a git repo" in ideas[0]["code_patch"]["reason"]


def test_code_patch_persistent_backend_rejects_target_repo_that_differs_from_snapshot(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_git_c2c_repo(tmp_path)
    snapshot = tmp_path / "snapshot"
    import shutil

    shutil.copytree(repo, snapshot, ignore=lambda directory, names: {".git"} & set(names))
    (repo / "rosetta/model/aligner.py").write_text("VALUE = 'not baseline'\n", encoding="utf-8")
    config = _code_patch_test_config(tmp_path / "workspace", snapshot)
    config["c2c"]["target_repo"] = str(repo)
    config["code_patch"]["backend"] = "codex_persistent_cli"
    config["code_patch"]["persistent_session"] = True
    config["code_patch"]["use_git_worktree"] = True
    paths = init_workspace(config, "topic", project_id="proj_baseline_guard", simulate=False)
    artifacts = ArtifactManager(paths.root)

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["worktree", "add"]:
            raise AssertionError("worktree add must not run when baseline guard fails")
        raise AssertionError(f"unexpected command: {command}")

    import auto_research.code_patch as code_patch_module

    monkeypatch.setattr(code_patch_module.subprocess, "run", fake_run)
    ideas = [{"id": "baseline_guard", "title": "Baseline Guard", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "no_valid_patch"
    assert "does not match the baseline snapshot" in ideas[0]["code_patch"]["reason"]


def test_code_patch_agent_filters_run_artifacts_from_patch_contract(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_contract_filter", simulate=False)
    artifacts = ArtifactManager(paths.root)
    captured: dict[str, object] = {}

    class InspectingBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            captured["implementation_contract"] = implementation_contract
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'contract filter'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Patched only editable files."}

    ideas = [
        {
            "id": "contract_filter",
            "title": "Contract Filter",
            "hypothesis": "h",
            "experiment_contract": {
                "expected_files": [
                    "rosetta/model/aligner.py",
                    "local/auto_research_runs/contract_filter/main_results.json",
                    "local/auto_research_runs/contract_filter/train_overrides.json",
                ]
            },
        }
    ]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=InspectingBackend()).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    contract = captured["implementation_contract"]
    assert isinstance(contract, dict)
    targets = contract["implementation_targets"]
    assert targets["expected_files"] == ["rosetta/model/aligner.py"]
    blocked_paths = {item["path"] for item in targets["blocked_expected_files"]}
    assert "local/auto_research_runs/contract_filter/main_results.json" in blocked_paths
    assert "local/auto_research_runs/contract_filter/train_overrides.json" in blocked_paths


def test_code_patch_contract_includes_c2c_mechanism_and_ablation(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_mechanism_contract", simulate=False)
    artifacts = ArtifactManager(paths.root)
    captured: dict[str, object] = {}

    class InspectingBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            captured.setdefault("implementation_contract", implementation_contract)
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'mechanism contract'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Patched mechanism target."}

    ideas = default_c2c_ideas("topic", {"name": "base", "mean": 50.0, "datasets": {}})
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=InspectingBackend()).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    contract = captured["implementation_contract"]
    assert isinstance(contract, dict)
    mechanism = contract["mechanism_contract"]
    assert mechanism["mechanism_type"] == "utility_predicted_cache_routing"
    assert mechanism["ablation_switch"] == "ablation_disable_utility_predicted_cache_routing"
    assert mechanism["coverage_diagnostics"]["required"] is True
    assert mechanism["matched_coverage_ablation"]["required"] is True
    assert mechanism["novelty_gate"]["status"] == "pass"
    scope = contract["implementation_scope"]
    assert scope["scope"] == "medium"
    assert "rosetta/model/utility_predicted_cache_routing.py" in scope["required_new_files"]
    assert scope["integration_points"]
    assert any("medium-scope" in requirement for requirement in contract["s2_5_requirements"])
    assert any("mechanism-level" in requirement for requirement in contract["s2_5_requirements"])
    assert any("matched-coverage" in requirement for requirement in contract["s2_5_requirements"])


def test_code_patch_contract_includes_s2_variant_fingerprint(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_variant_contract", simulate=False)
    artifacts = ArtifactManager(paths.root)
    captured: dict[str, object] = {}

    class InspectingBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            captured["implementation_contract"] = implementation_contract
            (temp_repo / "rosetta/model/wrapper.py").write_text("VALUE = 'variant contract'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Patched selected variant."}

    ideas = [
        {
            "id": "wrapper_arc_recovery_residual",
            "title": "Wrapper ARC recovery residual",
            "hypothesis": "h",
            "mechanism_type": "utility_predicted_cache_routing",
            "variant_fingerprint": "abc123variant",
            "s2_variant": {
                "variant_fingerprint": "abc123variant",
                "mechanism_axis": "normalization",
                "integration_point": "wrapper",
                "control_signal": "span_agreement",
                "variant_score": {"score": 5.0},
            },
            "experiment_contract": {
                "expected_files": ["rosetta/model/wrapper.py"],
                "ablation_switch": "ablation_disable_wrapper_arc_recovery_residual",
                "config_overrides": {
                    "train": {"model": {"cache_routing_mode": "wrapper_arc_recovery_residual"}},
                    "eval": {"model": {"rosetta_config": {"cache_routing_mode": "wrapper_arc_recovery_residual"}}},
                },
            },
        }
    ]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=InspectingBackend()).run({"candidate_ideas": ideas}, ideas)

    contract = captured["implementation_contract"]
    assert isinstance(contract, dict)
    assert contract["variant_fingerprint"] == "abc123variant"
    assert contract["s2_variant"]["integration_point"] == "wrapper"
    assert any("variant_fingerprint" in item for item in contract["s2_5_requirements"])
    assert manifest["selected_patch"]["variant_fingerprint"] == "abc123variant"
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))
    assert patch["variant_fingerprint"] == "abc123variant"
    assert patch["s2_variant"]["control_signal"] == "span_agreement"


def test_code_patch_contract_large_scope_uses_decomposition_guidance(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_large_scope_contract", simulate=False)
    artifacts = ArtifactManager(paths.root)
    captured: dict[str, object] = {}

    class InspectingBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            captured.setdefault("implementation_contract", implementation_contract)
            (temp_repo / "rosetta/model/projector.py").write_text("VALUE = 'large scope coherent slice'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Patched coherent mechanism slice."}

    ideas = default_c2c_ideas("topic", {"name": "base", "mean": 50.0, "datasets": {}})
    candidate = ideas[1]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=InspectingBackend()).run({"candidate_ideas": [candidate]}, [candidate])

    assert manifest["status"] == "ok"
    contract = captured["implementation_contract"]
    assert isinstance(contract, dict)
    assert contract["implementation_scope"]["scope"] == "large"
    assert contract["implementation_scope"]["mvp_slice"]
    assert any("large-scope" in requirement for requirement in contract["s2_5_requirements"])


def test_code_patch_agent_repairs_validation_failed_patch_once(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo, require_targeted_tests=True)
    paths = init_workspace(config, "topic", project_id="proj_patch_validation_repair", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class RepairingBackend:
        def __init__(self):
            self.calls = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.calls.append(implementation_contract)
            if len(self.calls) == 1:
                (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'needs repair'\n", encoding="utf-8")
                (temp_repo / "test/test_aligner_span_overlap.py").write_text(
                    "def test_span():\n    assert False\n",
                    encoding="utf-8",
                )
                return {"status": "ok", "rationale": "Initial patch has a failing focused test."}
            assert implementation_contract["validation_failure_feedback"]["failed_checks"]
            assert implementation_contract["s2_5_repair_session_policy"]["same_resume_session_required"] is True
            repair_packet = implementation_contract["codex_repair_packet"]
            assert repair_packet["repair_kind"] == "validation_or_activation_failure"
            assert repair_packet["failed_command_evidence"]
            assert repair_packet["changed_files"]
            (temp_repo / "test/test_aligner_span_overlap.py").write_text(
                "def test_span():\n    assert True\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Repaired the focused test failure."}

    backend = RepairingBackend()
    ideas = [{"id": "validation_repair", "title": "Validation Repair", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert len(backend.calls) == 2
    assert "validation_failure_feedback" in backend.calls[1]
    assert "codex_repair_packet" in backend.calls[1]
    assert validation["status"] == "ok"
    assert validation["recovery_actions"][0]["action"] == "retry_codex_after_validation_failure"
    assert validation["recovery_actions"][0]["repair_packet_summary"]["repair_kind"] == "validation_or_activation_failure"
    assert patch["recovery_actions"][0]["action"] == "retry_codex_after_validation_failure"


def test_code_patch_runtime_smoke_repairs_dtype_failure_before_proxy_train(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "script/train/SFT_train.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import json",
                "import os",
                "import sys",
                "if os.environ.get('WANDB_DISABLED') != 'true':",
                "    raise RuntimeError('wandb smoke must be disabled')",
                "config_path = Path(sys.argv[sys.argv.index('--config') + 1])",
                "cfg = json.loads(config_path.read_text(encoding='utf-8'))",
                "if cfg['data']['kwargs']['num_samples'] < 2:",
                "    raise RuntimeError('runtime smoke train sample split would be empty')",
                "if cfg['data']['train_ratio'] >= 0.99:",
                "    raise RuntimeError('runtime smoke train ratio was not hardened')",
                "aligner = Path('rosetta/model/aligner.py').read_text(encoding='utf-8')",
                "if 'runtime repaired' not in aligner:",
                "    raise RuntimeError('expected scalar type Float but found BFloat16')",
                "print('first batch ok')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["validation"]["runtime_smoke"] = {
        "enabled": True,
        "train_samples": 1,
        "timeout_seconds": 20,
        "gpu_ids": [],
        "mechanism_activation": {"enabled": False},
    }
    config["code_patch"]["validation"]["max_repair_attempts"] = 1
    paths = init_workspace(config, "topic", project_id="proj_patch_runtime_smoke", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class RuntimeRepairBackend:
        def __init__(self):
            self.calls = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.calls.append(implementation_contract)
            if len(self.calls) == 1:
                (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'runtime needs repair'\n", encoding="utf-8")
                return {"status": "ok", "rationale": "Initial patch leaves dtype mismatch in first batch."}
            feedback = implementation_contract["validation_failure_feedback"]
            assert feedback["failed_checks"][0]["name"] == "runtime_smoke:first_batch_train"
            assert feedback["failed_checks"][0]["failure_category"] == "dtype_mismatch"
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'runtime repaired'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Repaired dtype handling for first-batch smoke."}

    ideas = [{"id": "runtime_smoke_repair", "title": "Runtime Smoke Repair", "hypothesis": "h"}]
    backend = RuntimeRepairBackend()
    CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert ideas[0]["code_patch"]["status"] == "ok"
    assert len(backend.calls) == 2
    assert any(check["name"] == "runtime_smoke:first_batch_train" and check["returncode"] == 0 for check in validation["checks"])
    assert any(action["action"] == "retry_codex_after_validation_failure" for action in validation["recovery_actions"])


def test_code_patch_runtime_smoke_repairs_missing_mechanism_activation_wiring(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "script/train/SFT_train.py").write_text(
        "from pathlib import Path\n"
        "import json, os, sys\n"
        "config_path = Path(sys.argv[sys.argv.index('--config') + 1])\n"
        "cfg = json.loads(config_path.read_text(encoding='utf-8'))\n"
        "Path(cfg['output']['output_dir']).mkdir(parents=True, exist_ok=True)\n"
        "print('first batch ok')\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["validation"]["runtime_smoke"] = {
        "enabled": True,
        "train_samples": 2,
        "timeout_seconds": 20,
        "gpu_ids": [],
        "mechanism_activation": {"enabled": True, "hard_gate": True},
    }
    config["code_patch"]["validation"]["max_repair_attempts"] = 1
    paths = init_workspace(config, "topic", project_id="proj_patch_activation_wiring_smoke", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class WiringRepairBackend:
        def __init__(self):
            self.calls = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.calls.append(implementation_contract)
            if len(self.calls) == 1:
                (temp_repo / "rosetta/model/projector.py").write_text(
                    "VALUE = 'mechanism without switch wiring'\n",
                    encoding="utf-8",
                )
                return {"status": "ok", "rationale": "Initial patch forgets eval switch wiring."}
            feedback = implementation_contract["validation_failure_feedback"]
            assert feedback["failed_checks"][0]["name"] == "runtime_smoke:mechanism_activation_wiring"
            assert feedback["failed_checks"][0]["failure_category"] == "mechanism_activation_wiring_failed"
            (temp_repo / "rosetta/model/projector.py").write_text(
                "class Projector:\n"
                "    def forward(self, rosetta_config=None):\n"
                "        rosetta_config = rosetta_config or {}\n"
                "        if rosetta_config.get('disable_mechanism'):\n"
                "            return 'disabled'\n"
                "        return 'enabled'\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Repaired by reading ablation switch in projector forward path."}

    ideas = [
        {
            "id": "activation_wiring_repair",
            "title": "Activation Wiring Repair",
            "hypothesis": "h",
            "experiment_contract": {
                "ablation_switch": "disable_mechanism",
                "config_overrides": {"eval": {"model": {"rosetta_config": {"mechanism_enabled": True}}}},
            },
        }
    ]
    backend = WiringRepairBackend()
    CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert ideas[0]["code_patch"]["status"] == "validation_failed"
    assert len(backend.calls) == 2
    wiring_checks = [check for check in validation["checks"] if check["name"] == "runtime_smoke:mechanism_activation_wiring"]
    assert wiring_checks[-1]["status"] == "ok"
    assert wiring_checks[-1]["rosetta_config"]["disabled_switch_value"] is True
    assert wiring_checks[-1]["runtime_code_refs"]["switch_refs"] == ["rosetta/model/projector.py"]
    forward_checks = [check for check in validation["checks"] if check["name"] == "runtime_smoke:mechanism_activation_forward_probe"]
    assert forward_checks[-1]["status"] == "failed"
    assert forward_checks[-1]["blocking"] is True
    assert set(forward_checks[-1]["probe"]["failures"]) & {
        "enabled_disabled_forward_outputs_identical",
        "torch_import_failed",
        "projector_import_failed",
        "small_batch_forward_failed",
    }


def test_code_patch_runtime_smoke_uses_builtin_forward_probe_when_repo_script_missing(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "script/train/SFT_train.py").write_text(
        "from pathlib import Path\n"
        "import json, sys\n"
        "config_path = Path(sys.argv[sys.argv.index('--config') + 1])\n"
        "cfg = json.loads(config_path.read_text(encoding='utf-8'))\n"
        "Path(cfg['output']['output_dir']).mkdir(parents=True, exist_ok=True)\n"
        "print('first batch ok')\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["validation"]["runtime_smoke"] = {
        "enabled": True,
        "train_samples": 2,
        "timeout_seconds": 20,
        "gpu_ids": [],
        "mechanism_activation": {
            "enabled": True,
            "hard_gate": True,
            "forward_probe": {"enabled": True, "hard_gate": True, "timeout_seconds": 20},
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_patch_activation_forward_probe_builtin", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class ForwardProbeMissingBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/projector.py").write_text(
                "class Projector:\n"
                "    def forward(self, rosetta_config=None):\n"
                "        rosetta_config = rosetta_config or {}\n"
                "        if rosetta_config.get('disable_mechanism'):\n"
                "            return 'disabled'\n"
                "        return 'enabled'\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Patch wires the switch but the optional forward probe script is absent."}

    ideas = [
        {
            "id": "forward_probe_missing",
            "title": "Forward Probe Missing",
            "hypothesis": "h",
            "experiment_contract": {
                "ablation_switch": "disable_mechanism",
                "config_overrides": {"eval": {"model": {"rosetta_config": {"mechanism_enabled": True}}}},
            },
        }
    ]
    CodePatchAgent(paths.root, config, artifacts, backend=ForwardProbeMissingBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert ideas[0]["code_patch"]["status"] == "validation_failed"
    forward_checks = [check for check in validation["checks"] if check["name"] == "runtime_smoke:mechanism_activation_forward_probe"]
    assert forward_checks[-1]["status"] == "failed"
    assert forward_checks[-1]["returncode"] == 1
    assert forward_checks[-1]["probe_source"] == "builtin"
    assert forward_checks[-1]["probe_environment"]["probe_python"]
    assert "torch_available" in forward_checks[-1]["probe_environment"]
    assert "torch_version" in forward_checks[-1]["probe_environment"]
    assert "repo_import_ok" in forward_checks[-1]["probe_environment"]
    assert "using_c2c_env_python" in forward_checks[-1]["probe_environment"]
    assert forward_checks[-1]["probe"]["probe_type"] == "repo_small_batch_forward_failed_static_trace"
    assert forward_checks[-1]["probe"]["fallback_reason"] in {"torch_import_failed", "projector_import_failed", "small_batch_forward_failed"}


def test_code_patch_runtime_smoke_skips_forward_probe_when_no_script_and_no_builtin(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "script/train/SFT_train.py").write_text(
        "from pathlib import Path\n"
        "import json, sys\n"
        "config_path = Path(sys.argv[sys.argv.index('--config') + 1])\n"
        "cfg = json.loads(config_path.read_text(encoding='utf-8'))\n"
        "Path(cfg['output']['output_dir']).mkdir(parents=True, exist_ok=True)\n"
        "print('first batch ok')\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["validation"]["runtime_smoke"] = {
        "enabled": True,
        "train_samples": 2,
        "timeout_seconds": 20,
        "gpu_ids": [],
        "mechanism_activation": {
            "enabled": True,
            "hard_gate": True,
            "forward_probe": {"enabled": True, "hard_gate": True, "builtin_fallback": False, "timeout_seconds": 20},
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_patch_activation_forward_probe_missing", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class ForwardProbeMissingBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/projector.py").write_text(
                "class Projector:\n"
                "    def forward(self, rosetta_config=None):\n"
                "        rosetta_config = rosetta_config or {}\n"
                "        if rosetta_config.get('disable_mechanism'):\n"
                "            return 'disabled'\n"
                "        return 'enabled'\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Patch wires the switch but the optional forward probe script is absent."}

    ideas = [
        {
            "id": "forward_probe_missing",
            "title": "Forward Probe Missing",
            "hypothesis": "h",
            "experiment_contract": {
                "ablation_switch": "disable_mechanism",
                "config_overrides": {"eval": {"model": {"rosetta_config": {"mechanism_enabled": True}}}},
            },
        }
    ]
    CodePatchAgent(paths.root, config, artifacts, backend=ForwardProbeMissingBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert ideas[0]["code_patch"]["status"] == "ok"
    forward_checks = [check for check in validation["checks"] if check["name"] == "runtime_smoke:mechanism_activation_forward_probe"]
    assert forward_checks[-1]["status"] == "skipped"
    assert forward_checks[-1]["returncode"] == 0
    assert forward_checks[-1]["blocking"] is False


def test_code_patch_runtime_smoke_repairs_forward_probe_no_effect(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "script/auto_research").mkdir(parents=True, exist_ok=True)
    (repo / "script/auto_research/activation_forward_probe.py").write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--enabled-config')\n"
        "parser.add_argument('--disabled-config')\n"
        "parser.add_argument('--switch')\n"
        "parser.add_argument('--output')\n"
        "args = parser.parse_args()\n"
        "projector = Path('rosetta/model/projector.py').read_text(encoding='utf-8')\n"
        "observed = 'FORWARD_PROBE_ACTIVE_PATH' in projector\n"
        "payload = {\n"
        "    'compared_fields': ['projector_output_checksum'],\n"
        "    'changed_fields': ['projector_output_checksum'] if observed else [],\n"
        "    'unchanged_fields': [] if observed else ['projector_output_checksum'],\n"
        "    'mechanism_observed': observed,\n"
        "}\n"
        "open(args.output, 'w', encoding='utf-8').write(json.dumps(payload))\n",
        encoding="utf-8",
    )
    (repo / "script/train/SFT_train.py").write_text(
        "from pathlib import Path\n"
        "import json, sys\n"
        "config_path = Path(sys.argv[sys.argv.index('--config') + 1])\n"
        "cfg = json.loads(config_path.read_text(encoding='utf-8'))\n"
        "Path(cfg['output']['output_dir']).mkdir(parents=True, exist_ok=True)\n"
        "print('first batch ok')\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["validation"]["runtime_smoke"] = {
        "enabled": True,
        "train_samples": 2,
        "timeout_seconds": 20,
        "gpu_ids": [],
        "mechanism_activation": {
            "enabled": True,
            "hard_gate": True,
            "forward_probe": {"enabled": True, "hard_gate": True, "timeout_seconds": 20},
        },
    }
    config["code_patch"]["validation"]["max_repair_attempts"] = 1
    paths = init_workspace(config, "topic", project_id="proj_patch_activation_forward_probe_repair", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class ForwardProbeRepairBackend:
        def __init__(self):
            self.calls = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.calls.append(implementation_contract)
            (temp_repo / "rosetta/model/projector.py").write_text(
                "class Projector:\n"
                "    def forward(self, rosetta_config=None):\n"
                "        rosetta_config = rosetta_config or {}\n"
                "        if rosetta_config.get('disable_mechanism'):\n"
                "            return 'disabled'\n"
                "        return 'enabled'\n",
                encoding="utf-8",
            )
            if len(self.calls) == 1:
                return {"status": "ok", "rationale": "Initial patch wires config but leaves the probe unchanged."}
            feedback = implementation_contract["validation_failure_feedback"]
            failed = feedback["failed_checks"][0]
            assert failed["name"] == "runtime_smoke:mechanism_activation_forward_probe"
            assert failed["failure_category"] == "mechanism_activation_forward_probe_failed"
            assert "forward-level causal activation" in feedback["instruction"]
            assert any("Forward activation probe is mandatory" in requirement for requirement in implementation_contract["s2_5_requirements"])
            (temp_repo / "rosetta/model/projector.py").write_text(
                "FORWARD_PROBE_ACTIVE_PATH = True\n"
                "class Projector:\n"
                "    def forward(self, rosetta_config=None):\n"
                "        rosetta_config = rosetta_config or {}\n"
                "        if rosetta_config.get('disable_mechanism'):\n"
                "            return 'disabled'\n"
                "        return 'enabled_with_probe_effect'\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Repaired forward path so enabled/disabled probe observes a tensor change."}

    ideas = [
        {
            "id": "forward_probe_repair",
            "title": "Forward Probe Repair",
            "hypothesis": "h",
            "experiment_contract": {
                "ablation_switch": "disable_mechanism",
                "config_overrides": {"eval": {"model": {"rosetta_config": {"mechanism_enabled": True}}}},
            },
        }
    ]
    backend = ForwardProbeRepairBackend()
    CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert ideas[0]["code_patch"]["status"] == "ok"
    assert len(backend.calls) == 2
    forward_checks = [check for check in validation["checks"] if check["name"] == "runtime_smoke:mechanism_activation_forward_probe"]
    assert forward_checks[-1]["status"] == "ok"
    assert forward_checks[-1]["probe_source"] == "repo"
    assert forward_checks[-1]["probe"]["mechanism_observed"] is True
    assert forward_checks[-1]["probe"]["changed_fields"] == ["projector_output_checksum"]
    assert "script/auto_research/activation_forward_probe.py" not in ideas[0]["code_patch"]["changed_files"]
    assert any(action["action"] == "retry_codex_after_validation_failure" for action in validation["recovery_actions"])


def test_code_patch_repair_packet_includes_forward_probe_tensor_diagnostics(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "script/auto_research").mkdir(parents=True, exist_ok=True)
    (repo / "script/auto_research/activation_forward_probe.py").write_text(
        "import argparse, json\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--enabled-config')\n"
        "parser.add_argument('--disabled-config')\n"
        "parser.add_argument('--switch')\n"
        "parser.add_argument('--output')\n"
        "args = parser.parse_args()\n"
        "payload = {\n"
        "    'probe_type': 'repo_small_batch_forward',\n"
        "    'mechanism_observed': False,\n"
        "    'changed_fields': [],\n"
        "    'unchanged_fields': ['projector_output.key', 'wrapper_cache.layer0.key'],\n"
        "    'compared_fields': ['projector_output.key', 'wrapper_cache.layer0.key'],\n"
        "    'projector_called': True,\n"
        "    'switch_seen_by_forward': False,\n"
        "    'cache_key_diff': 0.0,\n"
        "    'cache_value_diff': 0.0,\n"
        "    'enabled': {'switch_value': None, 'rosetta_hash': 'enabled_hash'},\n"
        "    'disabled': {'switch_value': True, 'rosetta_hash': 'disabled_hash'},\n"
        "    'wrapper_probe': {'status': 'ok', 'projector_called': True, 'switch_seen_by_forward': False, 'cache_key_diff': 0.0, 'cache_value_diff': 0.0, 'failures': ['enabled_disabled_wrapper_cache_identical']},\n"
        "    'tensor_checks': [\n"
        "        {'name': 'projector_output.key', 'changed': False, 'max_abs_diff': 0.0, 'mean_abs_diff': 0.0, 'enabled_sha256': 'aaa', 'disabled_sha256': 'aaa', 'shape': [1, 2, 4, 4]},\n"
        "        {'name': 'wrapper_cache.layer0.key', 'changed': False, 'max_abs_diff': 0.0, 'mean_abs_diff': 0.0, 'enabled_sha256': 'bbb', 'disabled_sha256': 'bbb', 'shape': [1, 2, 4, 4]}\n"
        "    ],\n"
        "    'failures': ['enabled_disabled_wrapper_cache_identical']\n"
        "}\n"
        "open(args.output, 'w', encoding='utf-8').write(json.dumps(payload))\n",
        encoding="utf-8",
    )
    (repo / "script/train/SFT_train.py").write_text(
        "from pathlib import Path\n"
        "import json, sys\n"
        "config_path = Path(sys.argv[sys.argv.index('--config') + 1])\n"
        "cfg = json.loads(config_path.read_text(encoding='utf-8'))\n"
        "Path(cfg['output']['output_dir']).mkdir(parents=True, exist_ok=True)\n"
        "print('first batch ok')\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["validation"]["runtime_smoke"] = {
        "enabled": True,
        "train_samples": 2,
        "timeout_seconds": 20,
        "gpu_ids": [],
        "mechanism_activation": {
            "enabled": True,
            "hard_gate": True,
            "forward_probe": {"enabled": True, "hard_gate": True, "timeout_seconds": 20},
        },
    }
    config["code_patch"]["validation"]["max_repair_attempts"] = 1
    paths = init_workspace(config, "topic", project_id="proj_patch_forward_probe_tensor_packet", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class InspectForwardProbePacketBackend:
        def __init__(self):
            self.calls = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.calls.append(implementation_contract)
            (temp_repo / "rosetta/model/projector.py").write_text(
                "class Projector:\n"
                "    def forward(self, rosetta_config=None):\n"
                "        rosetta_config = rosetta_config or {}\n"
                "        if rosetta_config.get('disable_mechanism'):\n"
                "            return 'disabled_noop'\n"
                "        return 'enabled_noop'\n",
                encoding="utf-8",
            )
            if len(self.calls) == 1:
                return {"status": "ok", "rationale": "Initial no-op patch."}

            packet = implementation_contract["codex_repair_packet"]
            diagnostics = packet["activation_forward_probe_diagnostics"]
            assert diagnostics["probe_environment"]["probe_python"]
            assert "torch_available" in diagnostics["probe_environment"]
            assert "repo_import_ok" in diagnostics["probe_environment"]
            assert diagnostics["projector_called"] is True
            assert diagnostics["switch_seen_by_forward"] is False
            assert diagnostics["projector_output_identical"] is True
            assert diagnostics["wrapper_cache_identical"] is True
            assert diagnostics["identical_tensors"][0]["name"] == "projector_output.key"
            assert diagnostics["identical_tensors"][0]["enabled_sha256"] == "aaa"
            assert diagnostics["identical_tensors"][1]["disabled_sha256"] == "bbb"
            assert "forward_branch_missing_switch_or_rosetta_config_read" in diagnostics["repair_focus"]
            assert "constructor_params_or_projector_forward_branch_noop" in diagnostics["repair_focus"]
            assert "wrapper_cache_key_value_identical_enabled_disabled" in diagnostics["repair_focus"]
            feedback = implementation_contract["validation_failure_feedback"]
            failed = feedback["failed_checks"][0]
            assert failed["forward_probe_diagnostics"]["identical_tensors"][0]["enabled_sha256"] == "aaa"
            assert "activation_forward_probe_diagnostics" in feedback["instruction"]
            return {"status": "codex_failed", "reason": "stop after inspecting repair packet"}

    ideas = [
        {
            "id": "forward_probe_tensor_packet",
            "title": "Forward Probe Tensor Packet",
            "hypothesis": "h",
            "experiment_contract": {
                "ablation_switch": "disable_mechanism",
                "config_overrides": {"eval": {"model": {"rosetta_config": {"mechanism_enabled": True}}}},
            },
        }
    ]
    backend = InspectForwardProbePacketBackend()
    CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)

    assert len(backend.calls) == 2


def test_builtin_c2c_activation_forward_probe_detects_missing_forward_switch(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    enabled = repo / "enabled.yaml"
    disabled = repo / "disabled.yaml"
    enabled.write_text(
        yaml.safe_dump({"model": {"rosetta_config": {"mechanism_enabled": True}}}),
        encoding="utf-8",
    )
    disabled.write_text(
        yaml.safe_dump({"model": {"rosetta_config": {"mechanism_enabled": True, "disable_mechanism": True}}}),
        encoding="utf-8",
    )
    output = repo / "probe.json"
    probe = Path("src/auto_research/probes/c2c_activation_forward_probe.py").resolve()

    failed = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--enabled-config",
            "enabled.yaml",
            "--disabled-config",
            "disabled.yaml",
            "--switch",
            "disable_mechanism",
            "--output",
            str(output),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=20,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert failed.returncode == 1
    assert payload["mechanism_observed"] is False
    assert "runtime_forward_missing_switch_ref" in payload["failures"]

    (repo / "rosetta/model/projector.py").write_text(
        "class Projector:\n"
        "    def forward(self, rosetta_config=None):\n"
        "        rosetta_config = rosetta_config or {}\n"
        "        if rosetta_config.get('disable_mechanism'):\n"
        "            return 'disabled'\n"
        "        return 'enabled'\n",
        encoding="utf-8",
    )
    passed = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--enabled-config",
            "enabled.yaml",
            "--disabled-config",
            "disabled.yaml",
            "--switch",
            "disable_mechanism",
            "--output",
            str(output),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=20,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert passed.returncode == 1
    assert payload["mechanism_observed"] is False
    assert payload["probe_type"] == "repo_small_batch_forward_failed_static_trace"
    assert payload["trace"]["forward_refs"] == ["rosetta/model/projector.py"]


def test_builtin_c2c_activation_forward_probe_runs_repo_small_batch_forward(tmp_path: Path) -> None:
    if not _torch_available() or not _transformers_available():
        pytest.skip("torch and transformers are required for repo small-batch forward probe")
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/__init__.py").write_text("", encoding="utf-8")
    (repo / "rosetta/model/__init__.py").write_text("", encoding="utf-8")
    (repo / "rosetta/model/projector.py").write_text(
        "import torch\n"
        "class TinyProbeProjector(torch.nn.Module):\n"
        "    def __init__(self, source_dim, target_dim, disable_mechanism=False, **kwargs):\n"
        "        super().__init__()\n"
        "        self.disable_mechanism = disable_mechanism\n"
        "        self.scale = torch.nn.Parameter(torch.tensor(1.0))\n"
        "    def forward(self, source_kv, target_kv):\n"
        "        source_key, source_value = source_kv\n"
        "        target_key, target_value = target_kv\n"
        "        factor = 0.0 if self.disable_mechanism else self.scale\n"
        "        return target_key + factor * source_key, target_value + factor * source_value\n"
        "def create_projector(projector_type, **kwargs):\n"
        "    return TinyProbeProjector(**kwargs)\n",
        encoding="utf-8",
    )
    (repo / "rosetta/model/wrapper.py").write_text(
        "import torch\n"
        "class RosettaModel(torch.nn.Module):\n"
        "    def __init__(self, model_list, base_model_idx=0, projector_list=None, **kwargs):\n"
        "        super().__init__()\n"
        "        self.model_list = torch.nn.ModuleList(model_list)\n"
        "        self.base_model_idx = base_model_idx\n"
        "        self.projector_list = torch.nn.ModuleList(projector_list or [])\n"
        "        self.projector_dict = {}\n"
        "    def set_projector_config(self, source_model_idx, source_model_layer_idx, target_model_idx, target_model_layer_idx, projector_idx):\n"
        "        self.projector_dict[(source_model_idx, source_model_layer_idx, target_model_idx, target_model_layer_idx)] = projector_idx\n"
        "    def forward(self, input_ids=None, **kwargs):\n"
        "        del kwargs\n"
        "        base = self.model_list[0](input_ids=input_ids[:, :1], use_cache=True).past_key_values\n"
        "        source = self.model_list[1](input_ids=input_ids[:, :1], use_cache=True).past_key_values\n"
        "        projector = self.projector_list[0]\n"
        "        key, value = projector((source.key_cache[0], source.value_cache[0]), (base.key_cache[0], base.value_cache[0]))\n"
        "        base.key_cache[0] = key\n"
        "        base.value_cache[0] = value\n"
        "        return type('Output', (), {'past_key_values': base})()\n",
        encoding="utf-8",
    )
    enabled = repo / "enabled.yaml"
    disabled = repo / "disabled.yaml"
    enabled.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "rosetta_config": {
                        "mechanism_enabled": True,
                        "projector": {"type": "TinyProbeProjector", "params": {}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    disabled.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "rosetta_config": {
                        "mechanism_enabled": True,
                        "disable_mechanism": True,
                        "projector": {"type": "TinyProbeProjector", "params": {}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output = repo / "probe.json"
    probe = Path("src/auto_research/probes/c2c_activation_forward_probe.py").resolve()

    result = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--enabled-config",
            "enabled.yaml",
            "--disabled-config",
            "disabled.yaml",
            "--switch",
            "disable_mechanism",
            "--output",
            str(output),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=20,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert payload["probe_type"] == "repo_small_batch_forward"
    assert payload["mechanism_observed"] is True
    assert "projector_output.key" in payload["changed_fields"]
    assert "wrapper_cache.layer0.key" in payload["changed_fields"]
    assert payload["tensor_checks"][0]["max_abs_diff"] > 0
    assert payload["wrapper_probe"]["status"] == "ok"
    assert payload["wrapper_probe"]["projector_called"] is True
    assert payload["cache_key_diff"] > 0
    assert payload["cache_value_diff"] > 0
    assert payload["static_trace"]["mechanism_observed"] is False


def test_builtin_c2c_activation_forward_probe_fails_when_small_batch_forward_identical(tmp_path: Path) -> None:
    if not _torch_available():
        pytest.skip("torch is required for repo small-batch forward probe")
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/__init__.py").write_text("", encoding="utf-8")
    (repo / "rosetta/model/__init__.py").write_text("", encoding="utf-8")
    (repo / "rosetta/model/projector.py").write_text(
        "import torch\n"
        "class TinyProbeProjector(torch.nn.Module):\n"
        "    def __init__(self, source_dim, target_dim, disable_mechanism=False, **kwargs):\n"
        "        super().__init__()\n"
        "    def forward(self, source_kv, target_kv):\n"
        "        return target_kv\n"
        "def create_projector(projector_type, **kwargs):\n"
        "    return TinyProbeProjector(**kwargs)\n",
        encoding="utf-8",
    )
    enabled = repo / "enabled.yaml"
    disabled = repo / "disabled.yaml"
    enabled.write_text(
        yaml.safe_dump({"model": {"rosetta_config": {"projector": {"type": "TinyProbeProjector", "params": {}}}}}),
        encoding="utf-8",
    )
    disabled.write_text(
        yaml.safe_dump({"model": {"rosetta_config": {"disable_mechanism": True, "projector": {"type": "TinyProbeProjector", "params": {}}}}}),
        encoding="utf-8",
    )
    output = repo / "probe.json"
    probe = Path("src/auto_research/probes/c2c_activation_forward_probe.py").resolve()

    result = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--enabled-config",
            "enabled.yaml",
            "--disabled-config",
            "disabled.yaml",
            "--switch",
            "disable_mechanism",
            "--output",
            str(output),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=20,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result.returncode == 1
    assert payload["probe_type"] == "repo_small_batch_forward"
    assert payload["mechanism_observed"] is False
    assert "enabled_disabled_forward_tensors_identical" in payload["failures"]
    assert payload["unchanged_fields"] == ["projector_output.key", "projector_output.value"]


def test_forward_probe_normalization_trusts_explicit_wrapper_noop() -> None:
    evidence = _normalize_activation_forward_probe_payload(
        {
            "mechanism_observed": False,
            "changed_fields": ["projector_output.key"],
            "compared_fields": ["projector_output.key", "wrapper_cache.layer0.key"],
            "wrapper_probe": {
                "status": "ok",
                "projector_called": True,
                "cache_key_diff": 0.0,
                "cache_value_diff": 0.0,
            },
        },
        min_changed_fields=1,
    )

    assert evidence["mechanism_observed"] is False
    assert evidence["status"] == "unchanged"
    assert "enabled_disabled_forward_outputs_identical" in evidence["failures"]
    assert evidence["wrapper_probe"]["cache_key_diff"] == 0.0


def test_code_patch_validation_preserves_pytest_failure_context(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo, require_targeted_tests=True)
    config["code_patch"]["validation"]["max_repair_attempts"] = 0
    paths = init_workspace(config, "topic", project_id="proj_patch_failure_context", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class FailingTestBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'failure context'\n", encoding="utf-8")
            (temp_repo / "test/test_aligner_span_overlap.py").write_text(
                "def test_span():\n"
                "    assert 'actual boundary score' == 'expected boundary score'\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Patch with an intentional test failure."}

    ideas = [{"id": "failure_context", "title": "Failure Context", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=FailingTestBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "no_valid_patch"
    assert validation["status"] == "validation_failed"
    failed = [check for check in validation["checks"] if check["returncode"] != 0]
    assert failed
    assert "actual boundary score" in failed[0]["stdout"]
    assert "expected boundary score" in failed[0]["stdout"]


def test_code_patch_validation_records_pytest_timeout(tmp_path: Path, monkeypatch) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo, require_targeted_tests=True)
    config["code_patch"]["validation"]["max_repair_attempts"] = 0
    paths = init_workspace(config, "topic", project_id="proj_patch_pytest_timeout", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class TimeoutBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'timeout context'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Patch whose focused test times out."}

    import auto_research.code_patch as code_patch_module

    real_run = code_patch_module.subprocess.run

    def fake_run(command, *args, **kwargs):
        if "-m" in command and "pytest" in command:
            raise code_patch_module.subprocess.TimeoutExpired(command, kwargs.get("timeout"), output=b"partial pytest stdout", stderr=b"partial pytest stderr")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(code_patch_module.subprocess, "run", fake_run)

    ideas = [{"id": "pytest_timeout", "title": "Pytest Timeout", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=TimeoutBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "no_valid_patch"
    assert ideas[0]["code_patch"]["status"] == "validation_failed"
    timeout_check = next(check for check in validation["checks"] if check["name"].startswith("pytest:"))
    assert timeout_check["returncode"] == 124
    assert timeout_check["failure_category"] == "pytest_timeout"
    assert timeout_check["timeout_seconds"] == 180
    assert "partial pytest stdout" in timeout_check["stdout"]
    assert "Command timed out after 180s" in timeout_check["stderr"]


def test_code_patch_agent_marks_py_compile_failure_as_validation_failed(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_bad", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class BadBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/aligner.py").write_text("def broken(:\n", encoding="utf-8")
            return {"status": "ok", "rationale": "This patch has a syntax error."}

    ideas = [{"id": "bad_patch", "title": "Bad Patch", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=BadBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "no_valid_patch"
    assert ideas[0]["code_patch"]["status"] == "validation_failed"
    assert validation["status"] == "validation_failed"
    assert any(check["returncode"] != 0 for check in validation["checks"])


def test_code_patch_agent_blocks_unactivated_new_config_parameter_by_default(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/model/projector.py").write_text(
        "class C2CProjector:\n"
        "    def __init__(self, hidden_dim: int = 4):\n"
        "        self.hidden_dim = hidden_dim\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_activation_default_block", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class NewParamBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/projector.py").write_text(
                "class C2CProjector:\n"
                "    def __init__(\n"
                "        self,\n"
                "        hidden_dim: int = 4,\n"
                "        alignment_confidence_head_group_count: int = 2,\n"
                "    ):\n"
                "        self.hidden_dim = hidden_dim\n"
                "        self.alignment_confidence_head_group_count = alignment_confidence_head_group_count\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Added a configurable head group count."}

    ideas = [{"id": "unactivated_param", "title": "Unactivated Param", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=NewParamBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "no_valid_patch"
    assert ideas[0]["code_patch"]["status"] == "config_activation_missing"
    assert ideas[0]["code_patch"]["has_executable_change"] is False
    assert validation["activation_check"]["blocking"] is True
    assert validation["activation_check"]["missing_parameters"] == ["alignment_confidence_head_group_count"]
    assert validation["activation_check"]["blocking_missing_parameters"] == ["alignment_confidence_head_group_count"]


def test_code_patch_agent_can_track_unactivated_config_parameter_as_soft_debt(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/model/projector.py").write_text(
        "class C2CProjector:\n"
        "    def __init__(self, hidden_dim: int = 4):\n"
        "        self.hidden_dim = hidden_dim\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["validation"]["require_config_activation"] = "soft"
    paths = init_workspace(config, "topic", project_id="proj_patch_activation_soft_debt", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class NewParamBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/projector.py").write_text(
                "class C2CProjector:\n"
                "    def __init__(\n"
                "        self,\n"
                "        hidden_dim: int = 4,\n"
                "        alignment_confidence_head_group_count: int = 2,\n"
                "    ):\n"
                "        self.hidden_dim = hidden_dim\n"
                "        self.alignment_confidence_head_group_count = alignment_confidence_head_group_count\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Added a configurable head group count."}

    ideas = [{"id": "unactivated_param_soft", "title": "Unactivated Param Soft", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=NewParamBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert ideas[0]["code_patch"]["has_executable_change"] is True
    assert validation["activation_check"]["missing_parameters"] == ["alignment_confidence_head_group_count"]
    assert validation["activation_check"]["soft_issues"] == ["unactivated_config_parameter"]
    assert any(item["label"] == "unactivated_config_parameter" for item in ideas[0]["code_patch"]["quality_debt"])


def test_code_patch_agent_blocks_unactivated_new_config_parameter_in_strict_mode(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/model/projector.py").write_text(
        "class C2CProjector:\n"
        "    def __init__(self, hidden_dim: int = 4):\n"
        "        self.hidden_dim = hidden_dim\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["validation"]["gate_mode"] = "strict"
    paths = init_workspace(config, "topic", project_id="proj_patch_activation_strict", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class NewParamBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/projector.py").write_text(
                "class C2CProjector:\n"
                "    def __init__(\n"
                "        self,\n"
                "        hidden_dim: int = 4,\n"
                "        alignment_confidence_head_group_count: int = 2,\n"
                "    ):\n"
                "        self.hidden_dim = hidden_dim\n"
                "        self.alignment_confidence_head_group_count = alignment_confidence_head_group_count\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Added a configurable head group count."}

    ideas = [{"id": "unactivated_param_strict", "title": "Unactivated Param Strict", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=NewParamBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "no_valid_patch"
    assert ideas[0]["code_patch"]["status"] == "config_activation_missing"
    assert ideas[0]["code_patch"]["has_executable_change"] is False
    assert validation["activation_check"]["blocking"] is True
    assert validation["activation_check"]["missing_parameters"] == ["alignment_confidence_head_group_count"]


def test_code_patch_agent_repairs_unactivated_new_config_parameter(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/model/projector.py").write_text(
        "class C2CProjector:\n"
        "    def __init__(self, hidden_dim: int = 4):\n"
        "        self.hidden_dim = hidden_dim\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_activation_repair", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class RepairingParamBackend:
        def __init__(self):
            self.calls = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.calls.append(implementation_contract)
            if len(self.calls) == 1:
                (temp_repo / "rosetta/model/projector.py").write_text(
                    "class C2CProjector:\n"
                    "    def __init__(\n"
                    "        self,\n"
                    "        hidden_dim: int = 4,\n"
                    "        alignment_confidence_head_group_count: int = 2,\n"
                    "    ):\n"
                    "        self.hidden_dim = hidden_dim\n"
                    "        self.alignment_confidence_head_group_count = alignment_confidence_head_group_count\n",
                    encoding="utf-8",
                )
                return {"status": "ok", "rationale": "Added an unactivated configurable head group count."}
            feedback = implementation_contract["validation_failure_feedback"]
            assert feedback["activation_check"]["status"] == "config_activation_missing"
            assert feedback["activation_check"]["missing_parameters"] == ["alignment_confidence_head_group_count"]
            assert any("Config activation is mandatory" in requirement for requirement in implementation_contract["s2_5_requirements"])
            (temp_repo / "rosetta/model/projector.py").write_text(
                "class C2CProjector:\n"
                "    def __init__(self, hidden_dim: int = 4):\n"
                "        self.hidden_dim = hidden_dim\n"
                "        self.alignment_confidence_head_group_count = 2\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Repaired by keeping the value internal for the current implementation."}

    backend = RepairingParamBackend()
    ideas = [{"id": "activation_repair", "title": "Activation Repair", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert len(backend.calls) == 2
    assert validation["activation_check"]["status"] == "ok"
    assert validation["recovery_actions"][0]["action"] == "retry_codex_after_validation_failure"
    assert patch["recovery_actions"][0]["failed_checks"] == []


def test_code_patch_agent_accepts_new_config_parameter_when_contract_activates_it(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/model/projector.py").write_text(
        "class C2CProjector:\n"
        "    def __init__(self, hidden_dim: int = 4):\n"
        "        self.hidden_dim = hidden_dim\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_activation_ok", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class NewParamBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            assert implementation_contract["experiment_contract"]["config_overrides"]["train"]["model"]["projector"]["params"]["alignment_confidence_head_group_count"] == 2
            del edit_policy
            (temp_repo / "rosetta/model/projector.py").write_text(
                "class C2CProjector:\n"
                "    def __init__(\n"
                "        self,\n"
                "        hidden_dim: int = 4,\n"
                "        alignment_confidence_head_group_count: int = 2,\n"
                "    ):\n"
                "        self.hidden_dim = hidden_dim\n"
                "        self.alignment_confidence_head_group_count = alignment_confidence_head_group_count\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Added and activated a configurable head group count."}

    ideas = [
        {
            "id": "activated_param",
            "title": "Activated Param",
            "hypothesis": "h",
            "experiment_contract": {
                "config_overrides": {
                    "train": {
                        "model": {
                            "projector": {
                                "params": {
                                    "alignment_confidence_head_group_count": 2,
                                }
                            }
                        }
                    }
                }
            },
        }
    ]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=NewParamBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert validation["activation_check"]["activated_parameters"] == ["alignment_confidence_head_group_count"]


def test_code_patch_agent_ignores_standard_projector_list_constructor_param(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/model/wrapper.py").write_text(
        "class RosettaModel:\n"
        "    def __init__(self, model_list=None):\n"
        "        self.model_list = model_list\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_projector_list", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class ProjectorListBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/wrapper.py").write_text(
                "class RosettaModel:\n"
                "    def __init__(self, model_list=None, projector_list=None):\n"
                "        self.model_list = model_list\n"
                "        self.projector_list = projector_list or []\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Threaded an existing constructor dependency."}

    ideas = [{"id": "projector_list", "title": "Projector List", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=ProjectorListBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert validation["activation_check"]["status"] == "ok"
    assert validation["activation_check"]["introduced_config_parameters"] == []


def test_code_patch_agent_ignores_local_type_annotations_for_activation(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/model/aligner.py").write_text(
        "from typing import List, Tuple\n\n"
        "def score_spans():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_local_annotations", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class LocalAnnotationBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/aligner.py").write_text(
                "from typing import List, Tuple\n\n"
                "def score_spans():\n"
                "    selected_token_spans: List[Tuple[int, int]] = []\n"
                "    target_span: Tuple[int, int] = (0, 1)\n"
                "    selected_token_spans.append(target_span)\n"
                "    return len(selected_token_spans)\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Added local span bookkeeping."}

    ideas = [{"id": "local_annotations", "title": "Local annotations", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=LocalAnnotationBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert validation["activation_check"]["introduced_config_parameters"] == []


def test_code_patch_agent_includes_previous_patch_failure_in_retry_contract(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_retry_feedback", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class FailingBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/aligner.py").write_text("def broken(:\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Broken first attempt."}

    ideas = [{"id": "retry_feedback", "title": "Retry Feedback", "hypothesis": "h"}]
    first_manifest = CodePatchAgent(paths.root, config, artifacts, backend=FailingBackend()).run({"candidate_ideas": ideas}, ideas)
    assert first_manifest["status"] == "no_valid_patch"

    captured: dict[str, object] = {}

    class InspectingBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            captured["previous_patch_failure"] = implementation_contract.get("previous_patch_failure")
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'retry ok'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Retry used failure feedback."}

    retry_ideas = [{"id": "retry_feedback", "title": "Retry Feedback", "hypothesis": "h"}]
    retry_manifest = CodePatchAgent(paths.root, config, artifacts, backend=InspectingBackend()).run({"candidate_ideas": retry_ideas}, retry_ideas)

    assert retry_manifest["status"] == "ok"
    previous = captured["previous_patch_failure"]
    assert isinstance(previous, dict)
    assert previous["status"] == "validation_failed"
    assert previous["failed_checks"]


def test_code_patch_agent_includes_proxy_effect_repair_contract(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_proxy_effect_repair_feedback", simulate=False)
    artifacts = ArtifactManager(paths.root)
    results_dir = paths.root / "experiment" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    proxy_contract = {
        "mode": "effect_first_proxy_repair",
        "reason": "proxy mean delta 0.0 below soft threshold 0.1",
        "proxy_dataset_deltas": {"mmlu-redux": -0.2, "openbookqa": 0.1},
        "proxy_dataset_regressions": {"mmlu-redux": 0.2, "openbookqa": 0.0},
        "dragging_datasets": [{"dataset": "mmlu-redux", "delta": -0.2, "regression": 0.2}],
        "patch_risk_labels": ["test_change"],
        "repair_priorities": ["Target dragging proxy datasets: mmlu-redux"],
        "activation_smoke": {
            "status": "failed",
            "reason": "ablation switch produced no observable proxy eval metric or prediction change",
            "switch": "disable_mechanism",
            "datasets": ["mmlu-redux"],
            "eval_configs": {"mmlu-redux": "local/auto_research_runs/proxy_retry/proxy/activation_smoke_disabled/eval_mmlu-redux.yaml"},
            "enabled_metrics": {"mean": 50.5, "datasets": {"mmlu-redux": 50.5}},
            "disabled_metrics": {"mean": 50.5, "datasets": {"mmlu-redux": 50.5}},
            "metric_comparison": {"enabled_minus_disabled_mean": 0.0},
            "prediction_comparison": {"prediction_diff_rate": 0.0, "answer_diff_rate": 0.0},
        },
    }
    (results_dir / "main_results.json").write_text(
        json.dumps(
            {
                "candidate_results": [
                    {
                        "id": "proxy_retry",
                        "title": "Proxy Retry",
                        "decision": "proxy_repairable",
                        "patch_result": {"changed_files": ["rosetta/model/aligner.py"]},
                        "proxy_screen": {
                            "status": "repairable_proxy_risk",
                            "reason": proxy_contract["reason"],
                            "repair_hint": "effect repair only",
                            "proxy_effect_repair_contract": proxy_contract,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class InspectingBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            captured["implementation_contract"] = implementation_contract
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'proxy repair ok'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Used proxy evidence."}

    ideas = [{"id": "proxy_retry", "title": "Proxy Retry", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=InspectingBackend()).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    contract = captured["implementation_contract"]
    assert isinstance(contract, dict)
    proxy_effect = contract["proxy_effect_repair_contract"]
    assert isinstance(proxy_effect, dict)
    assert proxy_effect["mode"] == "effect_first_proxy_repair"
    assert proxy_effect["dragging_datasets"][0]["dataset"] == "mmlu-redux"
    requirements = contract["s2_5_requirements"]
    assert any("effect-first cheap-proxy repair" in requirement for requirement in requirements)
    assert any("mmlu-redux" in requirement for requirement in requirements)
    assert any("paperization-only" in requirement for requirement in requirements)
    assert any("Activation smoke failed" in requirement and "disable_mechanism" in requirement for requirement in requirements)
    assert any("prediction_comparison" in requirement for requirement in requirements)
    assert any("activation_smoke_disabled/eval_mmlu-redux.yaml" in requirement for requirement in requirements)


def test_code_patch_agent_marks_codex_429_as_retryable(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_429", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class RateLimitedBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, temp_repo, edit_policy
            return {
                "status": "codex_failed",
                "reason": "ERROR: exceeded retry limit, last status: 429 Too Many Requests",
                "stderr": "429 Too Many Requests",
            }

    ideas = [{"id": "rate_limited", "title": "Rate Limited", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=RateLimitedBackend()).run({"candidate_ideas": ideas}, ideas)
    plan_dir = paths.root / "plan"
    (plan_dir / "plan.yaml").write_text(
        yaml.safe_dump(
            {
                "hypotheses": [{"id": "h1"}],
                "baselines": [{"name": "base"}, {"name": "candidate"}],
                "datasets": [{"name": "mmlu-redux"}],
                "task_graph": {},
                "resource_budget": {},
                "execution": {
                    "collector": "c2c_small_loop",
                    "min_delta_to_pass": 0.1,
                    "max_dataset_regression": 2.0,
                },
                "acceptance_criteria": {
                    "minimum_mean_delta": 0.1,
                    "coverage_diagnostics_required": True,
                    "matched_coverage_ablation_required": True,
                },
                "ablation_matrix": [
                    {"experiment": "matched transfer coverage control", "matched_coverage_ablation": {"required": True}}
                ],
                "reviewer_risk_controls": {"top_concerns": []},
            }
        ),
        encoding="utf-8",
    )
    (plan_dir / "short_loop_plan.yaml").write_text("run: true\n", encoding="utf-8")
    (plan_dir / "candidate_ideas.json").write_text(json.dumps(default_c2c_ideas("topic", config["c2c"]["baseline"])), encoding="utf-8")
    _write_direction_and_variant_gate_artifacts(paths.root)
    report = S2GateValidator(paths.root, config).validate().to_dict()

    assert manifest["status"] == "retryable_no_valid_patch"
    assert manifest["retryable_patch_count"] == 1
    assert ideas[0]["code_patch"]["status"] == "retryable_codex_failed"
    assert ideas[0]["code_patch"]["retryable"] is True
    assert report["status"] == "NEEDS_RETRY"
    patch_status = next(check for check in report["checks"] if check["name"] == "s2_5_patch_manifest_status")
    assert patch_status["status"] == "NEEDS_RETRY"


def test_codex_retryable_detection_ignores_plain_quota_text() -> None:
    assert _codex_retryable_error_text("429 Too Many Requests") is True
    assert _codex_retryable_error_text("ERROR: insufficient_quota") is True
    assert _codex_retryable_error_text("Which sample type is called a quota sample?") is False
    assert _codex_retryable_error_text("apply_patch verification failed near quota sample text") is False


def test_runtime_smoke_gpu_attempts_auto_selects_most_free_gpu(monkeypatch) -> None:
    snapshot = [
        {"index": 0, "memory_total_mb": 24576, "memory_free_mb": 1024, "memory_used_mb": 23552, "utilization_gpu": 95},
        {"index": 1, "memory_total_mb": 24576, "memory_free_mb": 16000, "memory_used_mb": 8576, "utilization_gpu": 5},
        {"index": 2, "memory_total_mb": 24576, "memory_free_mb": 12000, "memory_used_mb": 12576, "utilization_gpu": 0},
    ]
    monkeypatch.setattr("auto_research.adapters.runner.ExperimentRunner._gpu_snapshot", staticmethod(lambda: snapshot))

    attempts = _runtime_smoke_gpu_attempts(
        {"experiment": {"gpu_policy": {"max_gpus": 6}}},
        {"gpu_ids": "auto", "min_free_mb": 8192, "resource_wait": {"enabled": False}},
    )

    assert attempts[0]["selected_gpu_ids"] == [1]
    assert attempts[0]["memory_free_mb"] == 16000
    assert attempts[0]["reason"] == "runtime_smoke_auto_selected_by_free_memory"


def test_c2c_runtime_smoke_ignores_legacy_fixed_gpu_by_default() -> None:
    config = {
        "c2c": {"enabled": True},
        "code_patch": {
            "validation": {
                "runtime_smoke": {
                    "gpu_ids": [0],
                    "min_free_mb": 8192,
                }
            }
        },
    }

    smoke_cfg = _code_patch_config(config)["validation"]["runtime_smoke"]

    assert smoke_cfg["gpu_ids"] == "auto"
    assert smoke_cfg["gpu_selection_policy"] == "auto_free_memory"
    assert smoke_cfg["legacy_configured_gpu_ids"] == [0]


def test_c2c_runtime_smoke_can_respect_explicit_fixed_gpu_opt_in() -> None:
    config = {
        "c2c": {"enabled": True},
        "code_patch": {
            "validation": {
                "runtime_smoke": {
                    "gpu_ids": [0],
                    "respect_configured_gpu_ids": True,
                }
            }
        },
    }

    smoke_cfg = _code_patch_config(config)["validation"]["runtime_smoke"]

    assert smoke_cfg["gpu_ids"] == [0]
    assert "legacy_configured_gpu_ids" not in smoke_cfg


def test_runtime_smoke_gpu_attempts_resource_unavailable_waits_then_pauses(monkeypatch) -> None:
    snapshot = [
        {"index": 0, "memory_total_mb": 24576, "memory_free_mb": 1024, "memory_used_mb": 23552, "utilization_gpu": 95},
        {"index": 1, "memory_total_mb": 24576, "memory_free_mb": 4096, "memory_used_mb": 20480, "utilization_gpu": 5},
    ]
    monkeypatch.setattr("auto_research.adapters.runner.ExperimentRunner._gpu_snapshot", staticmethod(lambda: snapshot))

    attempts = _runtime_smoke_gpu_attempts(
        {"experiment": {"gpu_policy": {"max_gpus": 6}}},
        {"gpu_ids": "auto", "min_free_mb": 8192, "resource_wait": {"enabled": True, "timeout_seconds": 0, "poll_seconds": 1}},
    )

    assert attempts[0]["resource_unavailable"] is True
    assert attempts[0]["reason"] == "runtime_smoke_resource_wait_timeout"
    assert attempts[0]["selected_gpu_ids"] == [1]
    assert attempts[0]["resource_wait"]["status"] == "timeout"


def test_runtime_smoke_oom_retry_switches_to_untried_free_gpu(monkeypatch) -> None:
    snapshot = [
        {"index": 0, "memory_total_mb": 24576, "memory_free_mb": 16000, "memory_used_mb": 8576, "utilization_gpu": 0},
        {"index": 1, "memory_total_mb": 24576, "memory_free_mb": 15000, "memory_used_mb": 9576, "utilization_gpu": 0},
    ]
    monkeypatch.setattr("auto_research.adapters.runner.ExperimentRunner._gpu_snapshot", staticmethod(lambda: snapshot))

    retry = _runtime_smoke_oom_retry_attempt(
        {"experiment": {"gpu_policy": {"max_gpus": 6}}},
        {"gpu_ids": "auto", "min_free_mb": 8192, "resource_wait": {"enabled": False}},
        tried_gpu_ids={0},
    )

    assert retry is not None
    assert retry["selected_gpu_ids"] == [1]
    assert retry["reason"] == "runtime_smoke_oom_retry_auto_selected_by_free_memory"


def test_patch_failure_retryable_treats_runtime_smoke_resource_retry_as_retryable() -> None:
    entry = {
        "status": "validation_failed",
        "validation": {
            "status": "validation_failed",
            "checks": [
                {
                    "name": "runtime_smoke:first_batch_train",
                    "returncode": 75,
                    "failure_category": "runtime_smoke_resource_retry",
                    "resource_retry": True,
                }
            ],
        },
    }

    assert _patch_failure_retryable(entry) is True


def test_code_patch_agent_repairs_blocked_no_executable_change(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_noop_contract_repair", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class NoopThenPatchBackend:
        def __init__(self):
            self.calls = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.calls.append(implementation_contract)
            if len(self.calls) == 1:
                return {"status": "ok", "rationale": "Initial attempt did not edit files."}
            feedback = implementation_contract["contract_failure_feedback"]
            assert feedback["status"] == "blocked_no_executable_change"
            assert "no allowed file changes" in feedback["reason"]
            assert implementation_contract["s2_5_repair_session_policy"]["same_resume_session_required"] is True
            repair_packet = implementation_contract["codex_repair_packet"]
            assert repair_packet["repair_kind"] == "contract_failure"
            assert repair_packet["failed_status"] == "blocked_no_executable_change"
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'contract repaired'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Repaired by editing an allowed integration point."}

    backend = NoopThenPatchBackend()
    ideas = [{"id": "noop_contract_repair", "title": "Noop Contract Repair", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert manifest["valid_patch_count"] == 1
    assert manifest["selected_candidate_id"] == "noop_contract_repair"
    assert manifest["selected_patch"]["candidate_id"] == "noop_contract_repair"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert len(backend.calls) == 2
    assert "codex_repair_packet" in backend.calls[1]
    assert validation["status"] == "ok"
    assert patch["recovery_actions"][0]["action"] == "retry_codex_after_contract_failure"
    assert patch["recovery_actions"][0]["failed_status"] == "blocked_no_executable_change"
    assert patch["recovery_actions"][0]["repair_packet_summary"]["repair_kind"] == "contract_failure"


def test_code_patch_agent_repairs_selected_variant_after_validation_failure(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 2
    config["code_patch"]["validation"]["max_repair_attempts"] = 1
    config["code_patch"]["validation"]["max_contract_repair_attempts"] = 0
    paths = init_workspace(config, "topic", project_id="proj_single_variant_repair", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [{"id": "single_variant_repair", "title": "Single Variant Repair", "hypothesis": "h"}]

    class RepairBackend:
        def __init__(self):
            self.contracts = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.contracts.append(implementation_contract)
            if len(self.contracts) == 1:
                (temp_repo / "rosetta/model/aligner.py").write_text("def broken(:\n", encoding="utf-8")
                return {"status": "ok", "rationale": "broken initial implementation"}
            assert "validation_failure_feedback" in implementation_contract
            assert "patch_variant" not in implementation_contract
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'validation repaired'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "valid repair for the same selected variant"}

    backend = RepairBackend()
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    assert manifest["selection_policy"]["mode"] == "single_s2_selected_variant"
    assert manifest["selection_policy"]["ignored_legacy_config"]["variants_per_candidate"] == 2
    assert len(backend.contracts) == 2
    assert manifest["selected_candidate_id"] == "single_variant_repair"
    assert manifest["selected_patch"]["selected_variant"] == 1
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert ideas[0]["code_patch"]["selected_variant"] == 1
    assert ideas[0]["code_patch"]["selection_reason"] == "single_s2_selected_variant"
    assert ideas[0]["code_patch"]["patch_json"].endswith("plan/code_patches/single_variant_repair/patch.json")


def test_code_patch_agent_does_not_best_of_n_score_ok_variants(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 2
    config["code_patch"]["stop_after_first_ok_score"] = 0
    config["code_patch"]["validation"]["max_repair_attempts"] = 0
    config["code_patch"]["validation"]["mechanism_self_review"] = {"enabled": True}
    paths = init_workspace(config, "topic", project_id="proj_no_best_quality", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [
        {
            "id": "quality_single",
            "title": "Quality Single",
            "hypothesis": "h",
            "mechanism_type": "utility_predicted_cache_routing",
            "ablation_plan": {"switch": "ablation_disable_quality_single"},
            "coverage_diagnostics": {"required": True},
            "matched_coverage_ablation": {"required": True},
            "experiment_contract": {
                "ablation_switch": "ablation_disable_quality_single",
                "coverage_diagnostics": {"required": True},
                "matched_coverage_ablation": {"required": True},
                "expected_files": ["rosetta/model/projector.py"],
            },
        }
    ]

    class QualityBackend:
        def __init__(self):
            self.calls = 0

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            self.calls += 1
            (temp_repo / "rosetta/model/projector.py").write_text(
                "QUALITY = 'single ok implementation without a second scored variant'\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "First valid implementation is kept; S2.5 repair handles failures."}

    backend = QualityBackend()
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    assert backend.calls == 1
    assert manifest["selected_candidate_id"] == "quality_single"
    assert manifest["selected_patch"]["selected_variant"] == 1
    assert manifest["selected_patch"]["quality_score"]["score"] > 0
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert ideas[0]["code_patch"]["selected_variant"] == 1
    assert ideas[0]["code_patch"]["selection_reason"] == "single_s2_selected_variant"
    attempts = ideas[0]["code_patch"]["variant_attempts"]
    assert len(attempts) == 1
    assert attempts[0]["status"] == "ok"
    assert "ablation_switch_not_wired" in attempts[0]["quality_score"]["soft_issues"]
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))
    assert validation["mechanism_review"]["status"] == "ok"
    assert validation["mechanism_review"]["mechanism_evidence_map"]


def test_code_patch_agent_ignores_legacy_variant_budget_after_first_ok_patch(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 2
    config["code_patch"]["stop_after_first_ok_score"] = 100
    config["code_patch"]["validation"]["max_repair_attempts"] = 0
    config["code_patch"]["validation"]["mechanism_self_review"] = {"enabled": True}
    paths = init_workspace(config, "topic", project_id="proj_best_quality_early_stop", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [
        {
            "id": "early_stop_quality",
            "title": "Early Stop Quality",
            "hypothesis": "h",
            "mechanism_type": "utility_predicted_cache_routing",
            "ablation_plan": {"switch": "ablation_disable_early_stop_quality"},
            "coverage_diagnostics": {"required": True},
            "matched_coverage_ablation": {"required": True},
            "experiment_contract": {
                "ablation_switch": "ablation_disable_early_stop_quality",
                "coverage_diagnostics": {"required": True},
                "matched_coverage_ablation": {"required": True},
                "expected_files": ["rosetta/model/projector.py"],
            },
        }
    ]

    class HighQualityBackend:
        def __init__(self):
            self.calls = 0

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            self.calls += 1
            (temp_repo / "rosetta/model/projector.py").write_text(
                "\n".join(
                    [
                        "ablation_disable_early_stop_quality = False",
                        "def route_cache(control_mode='matched_coverage_early_stop_quality'):",
                        "    coverage_diagnostics = {'accepted_span_rate': 0.5}",
                        "    matched_coverage_delta = 0.0",
                        "    return coverage_diagnostics, matched_coverage_delta",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "First variant is already high quality."}

    backend = HighQualityBackend()
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    assert backend.calls == 1
    assert manifest["selected_candidate_id"] == "early_stop_quality"
    assert manifest["selected_patch"]["selected_variant"] == 1
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert ideas[0]["code_patch"]["selected_variant"] == 1
    assert len(ideas[0]["code_patch"]["variant_attempts"]) == 1
    assert ideas[0]["code_patch"]["quality_score"]["score"] >= 100


def test_code_patch_agent_skips_later_candidates_after_high_quality_patch(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["max_candidates"] = 2
    config["code_patch"]["variants_per_candidate"] = 1
    config["code_patch"]["stop_after_first_ok_score"] = 100
    config["code_patch"]["validation"]["max_repair_attempts"] = 0
    config["code_patch"]["validation"]["mechanism_self_review"] = {"enabled": True}
    paths = init_workspace(config, "topic", project_id="proj_candidate_early_stop", simulate=False)
    artifacts = ArtifactManager(paths.root)
    first = {
        "id": "candidate_winner",
        "title": "Candidate Winner",
        "hypothesis": "h",
        "mechanism_type": "utility_predicted_cache_routing",
        "ablation_plan": {"switch": "ablation_disable_candidate_winner"},
        "coverage_diagnostics": {"required": True},
        "matched_coverage_ablation": {"required": True},
        "experiment_contract": {
            "ablation_switch": "ablation_disable_candidate_winner",
            "coverage_diagnostics": {"required": True},
            "matched_coverage_ablation": {"required": True},
            "expected_files": ["rosetta/model/projector.py"],
        },
    }
    second = {"id": "candidate_should_skip", "title": "Candidate Should Skip", "hypothesis": "h"}
    ideas = [first, second]

    class HighQualityBackend:
        def __init__(self):
            self.calls = 0

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            self.calls += 1
            (temp_repo / "rosetta/model/projector.py").write_text(
                "\n".join(
                    [
                        "ablation_disable_candidate_winner = False",
                        "def route_cache(control_mode='matched_coverage_candidate_winner'):",
                        "    coverage_diagnostics = {'accepted_span_rate': 0.5}",
                        "    matched_coverage_delta = 0.0",
                        "    return coverage_diagnostics, matched_coverage_delta",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "First candidate is enough for effect-first discovery."}

    backend = HighQualityBackend()
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    assert manifest["selection_policy"]["mode"] == "single_s2_selected_variant"
    assert backend.calls == 1
    assert manifest["candidate_count"] == 1
    assert manifest["selected_candidate_id"] == "candidate_winner"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert ideas[1]["code_patch"]["status"] == "skipped_s2_5_single_candidate_mode"


def test_code_patch_agent_implements_only_plan_selected_s2_variant(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["max_candidates"] = 3
    config["code_patch"]["variants_per_candidate"] = 2
    paths = init_workspace(config, "topic", project_id="proj_selected_s2_variant_only", simulate=False)
    artifacts = ArtifactManager(paths.root)
    first = {
        "id": "unselected_first",
        "title": "Unselected First",
        "hypothesis": "h",
        "variant_fingerprint": "fp_unselected",
    }
    second = {
        "id": "selected_second",
        "title": "Selected Second",
        "hypothesis": "h",
        "variant_fingerprint": "fp_selected",
    }
    third = {
        "id": "unselected_third",
        "title": "Unselected Third",
        "hypothesis": "h",
        "variant_fingerprint": "fp_third",
    }
    ideas = [first, second, third]

    class SelectedBackend:
        def __init__(self):
            self.contracts = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.contracts.append(implementation_contract)
            assert implementation_contract["candidate_id"] == "selected_second"
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'selected second only'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Implemented S2-selected variant."}

    backend = SelectedBackend()
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=backend).run(
        {
            "candidate_ideas": ideas,
            "selected_idea": second,
            "selected_variant_candidates": [{"id": "selected_second", "variant_fingerprint": "fp_selected"}],
        },
        ideas,
    )

    assert manifest["status"] == "ok"
    assert manifest["selection_policy"]["selected_by"] == "plan.selected_idea"
    assert manifest["selected_candidate_id"] == "selected_second"
    assert manifest["candidate_count"] == 1
    assert manifest["input_candidate_count"] == 3
    assert manifest["skipped_candidate_count"] == 2
    assert len(backend.contracts) == 1
    assert first["code_patch"]["status"] == "skipped_s2_5_single_candidate_mode"
    assert second["code_patch"]["status"] == "ok"
    assert third["code_patch"]["status"] == "skipped_s2_5_single_candidate_mode"


def test_code_patch_mechanism_self_review_keeps_diagnostics_soft_by_default(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 1
    config["code_patch"]["validation"]["max_repair_attempts"] = 1
    config["code_patch"]["validation"]["mechanism_self_review"] = {"enabled": True}
    paths = init_workspace(config, "topic", project_id="proj_mechanism_review_soft", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [
        {
            "id": "review_repair",
            "title": "Review Repair",
            "hypothesis": "h",
            "mechanism_type": "utility_predicted_cache_routing",
            "ablation_plan": {"switch": "ablation_disable_review_repair"},
            "coverage_diagnostics": {"required": True},
            "matched_coverage_ablation": {"required": True},
            "experiment_contract": {
                "ablation_switch": "ablation_disable_review_repair",
                "coverage_diagnostics": {"required": True},
                "matched_coverage_ablation": {"required": True},
                "expected_files": ["rosetta/model/projector.py"],
            },
        }
    ]

    class SoftReviewBackend:
        def __init__(self):
            self.contracts = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.contracts.append(implementation_contract)
            if len(self.contracts) == 1:
                (temp_repo / "rosetta/model/projector.py").write_text(
                    "VALUE = 'mechanism shell without diagnostics'\n",
                    encoding="utf-8",
                )
                return {"status": "ok", "rationale": "Missing ablation and diagnostics."}

    backend = SoftReviewBackend()
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert len(backend.contracts) == 1
    assert validation["mechanism_review"]["status"] == "ok"
    assert "ablation_switch_not_wired" in validation["mechanism_review"]["soft_issues"]
    assert validation["mechanism_review"]["quality_repair"]["needed"] is False
    assert validation["mechanism_review"]["quality_repair"]["deferred"] is True
    assert validation["mechanism_review"]["quality_repair"]["mode"] == "paperization_after_effect"
    assert ideas[0]["code_patch"]["quality_score"]["soft_issues"]


def test_code_patch_mechanism_self_review_can_be_strict_for_diagnostics(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 1
    config["code_patch"]["validation"]["max_repair_attempts"] = 0
    config["code_patch"]["validation"]["gate_mode"] = "strict"
    config["code_patch"]["validation"]["mechanism_self_review"] = {
        "enabled": True,
        "require_ablation_wired": True,
        "require_coverage_evidence": True,
        "require_matched_coverage_evidence": True,
    }
    paths = init_workspace(config, "topic", project_id="proj_mechanism_review_strict", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [
        {
            "id": "strict_review",
            "title": "Strict Review",
            "hypothesis": "h",
            "mechanism_type": "utility_predicted_cache_routing",
            "ablation_plan": {"switch": "ablation_disable_strict_review"},
            "coverage_diagnostics": {"required": True},
            "matched_coverage_ablation": {"required": True},
            "experiment_contract": {
                "ablation_switch": "ablation_disable_strict_review",
                "coverage_diagnostics": {"required": True},
                "matched_coverage_ablation": {"required": True},
                "expected_files": ["rosetta/model/projector.py"],
            },
        }
    ]

    class StrictBackend:
            def generate(self, implementation_contract, temp_repo, edit_policy):
                del implementation_contract, edit_policy
                (temp_repo / "rosetta/model/projector.py").write_text(
                    "VALUE = 'mechanism shell only'\n",
                    encoding="utf-8",
                )
                return {"status": "ok", "rationale": "Missing strict diagnostics."}

    manifest = CodePatchAgent(paths.root, config, artifacts, backend=StrictBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "no_valid_patch"
    assert ideas[0]["code_patch"]["status"] == "mechanism_self_review_failed"
    assert "ablation_switch_not_wired" in validation["mechanism_review"]["issues"]
    assert "missing_coverage_diagnostics_evidence" in validation["mechanism_review"]["issues"]


def test_code_patch_agent_repairs_evaluator_proxy_risk(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 1
    config["code_patch"]["validation"]["auto_prune_scope"] = False
    paths = init_workspace(config, "topic", project_id="proj_eval_repair", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [{"id": "eval_repair", "title": "Eval Repair", "hypothesis": "h"}]

    class EvalRiskBackend:
        def __init__(self):
            self.contracts = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.contracts.append(implementation_contract)
            if len(self.contracts) == 1:
                (temp_repo / "script/evaluation/unified_evaluator.py").write_text("print('contaminated')\n", encoding="utf-8")
                return {"status": "ok", "rationale": "bad evaluator hook"}
            (temp_repo / "script/evaluation/unified_evaluator.py").write_text("print('eval')\n", encoding="utf-8")
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'repaired mechanism'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "safe repair"}

    backend = EvalRiskBackend()
    CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))

    assert ideas[0]["code_patch"]["status"] == "ok"
    assert len(backend.contracts) == 2
    assert backend.contracts[1]["contract_failure_feedback"]["status"] == "proxy_risk_repair_required"
    assert patch["changed_files"] == ["rosetta/model/aligner.py"]
    assert validation["risk_check"]["status"] == "ok"
    assert any(action["failed_status"] == "proxy_risk_repair_required" for action in validation["recovery_actions"])
    assert any(action["action"] == "restore_evaluator_files_before_repair" for action in validation["recovery_actions"])


def test_code_patch_soft_allows_over_scope_file_count_by_default(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/model/helper_a.py").write_text("VALUE = 'a'\n", encoding="utf-8")
    (repo / "rosetta/model/helper_b.py").write_text("VALUE = 'b'\n", encoding="utf-8")
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 1
    config["code_patch"]["validation"]["max_changed_files"] = 2
    paths = init_workspace(config, "topic", project_id="proj_soft_over_scope", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [{"id": "soft_over_scope", "title": "Soft Over Scope", "hypothesis": "h"}]

    class BroadButSafeBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'aligner patch'\n", encoding="utf-8")
            (temp_repo / "rosetta/model/projector.py").write_text("VALUE = 'projector patch'\n", encoding="utf-8")
            (temp_repo / "rosetta/model/helper_a.py").write_text("VALUE = 'helper a patch'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Broad but safe mechanism patch."}

    manifest = CodePatchAgent(paths.root, config, artifacts, backend=BroadButSafeBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert len(patch["changed_files"]) == 3
    assert validation["risk_check"]["status"] == "ok"
    assert "patch_too_broad" in validation["risk_check"]["risk_labels"]
    assert validation["risk_check"]["warnings"]
    assert "patch_too_broad" in patch["quality_score"]["risk_labels"]


def test_code_patch_strict_max_changed_files_still_blocks(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/model/helper_a.py").write_text("VALUE = 'a'\n", encoding="utf-8")
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 1
    config["code_patch"]["validation"]["max_changed_files"] = 2
    config["code_patch"]["validation"]["strict_max_changed_files"] = True
    config["code_patch"]["validation"]["auto_prune_over_scope_files"] = False
    paths = init_workspace(config, "topic", project_id="proj_strict_over_scope", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [{"id": "strict_over_scope", "title": "Strict Over Scope", "hypothesis": "h"}]

    class BroadBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'aligner patch'\n", encoding="utf-8")
            (temp_repo / "rosetta/model/projector.py").write_text("VALUE = 'projector patch'\n", encoding="utf-8")
            (temp_repo / "rosetta/model/helper_a.py").write_text("VALUE = 'helper a patch'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Over strict file budget."}

    manifest = CodePatchAgent(paths.root, config, artifacts, backend=BroadBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "no_valid_patch"
    assert ideas[0]["code_patch"]["status"] == "patch_too_broad"
    assert validation["risk_check"]["status"] == "patch_too_broad"
    assert validation["risk_check"]["risk_labels"] == ["patch_too_broad"]


def test_code_patch_agent_blocks_evaluator_repair_that_recontaminates(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 1
    config["code_patch"]["validation"]["max_contract_repair_attempts"] = 1
    config["code_patch"]["validation"]["auto_prune_scope"] = False
    paths = init_workspace(config, "topic", project_id="proj_eval_repair_recontaminate", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [{"id": "eval_recontaminate", "title": "Eval Recontaminate", "hypothesis": "h"}]

    class RecontaminatingBackend:
        def __init__(self):
            self.contracts = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.contracts.append(implementation_contract)
            (temp_repo / "script/evaluation/unified_evaluator.py").write_text("print('still contaminated')\n", encoding="utf-8")
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'mechanism attempt'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "still touched evaluator"}

    backend = RecontaminatingBackend()
    CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert ideas[0]["code_patch"]["status"] == "proxy_risk_repair_required"
    assert len(backend.contracts) == 2
    assert backend.contracts[1]["forbidden_repair_files"] == ["script/evaluation/"]
    assert validation["risk_check"]["risk_labels"] == ["evaluation_code_changed"]
    assert any(action["action"] == "restore_evaluator_files_before_repair" for action in validation["recovery_actions"])


def test_code_patch_auto_prunes_evaluator_and_low_priority_over_scope_files(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/train").mkdir(parents=True, exist_ok=True)
    (repo / "rosetta/train/model_utils.py").write_text("VALUE = 'model utils'\n", encoding="utf-8")
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 1
    config["code_patch"]["validation"]["max_changed_files"] = 4
    config["code_patch"]["validation"]["auto_prune_over_scope_files"] = True
    paths = init_workspace(config, "topic", project_id="proj_auto_prune_scope", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [
        {
            "id": "auto_prune_scope",
            "title": "Auto Prune Scope",
            "hypothesis": "h",
            "experiment_contract": {
                "expected_files": [
                    "rosetta/model/aligner.py",
                    "rosetta/model/projector.py",
                    "rosetta/model/auto_prune_scope.py",
                    "test/test_aligner_span_overlap.py",
                ]
            },
            "implementation_plan": {
                "required_new_files": ["rosetta/model/auto_prune_scope.py"],
                "smoke_tests": ["test/test_aligner_span_overlap.py"],
            },
        }
    ]

    class BroadBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'aligner mechanism'\n", encoding="utf-8")
            (temp_repo / "rosetta/model/projector.py").write_text("VALUE = 'projector mechanism'\n", encoding="utf-8")
            (temp_repo / "rosetta/model/auto_prune_scope.py").write_text("VALUE = 'new mechanism'\n", encoding="utf-8")
            (temp_repo / "test/test_aligner_span_overlap.py").write_text("def test_scope():\n    assert True\n", encoding="utf-8")
            (temp_repo / "script/train/SFT_train.py").write_text("print('broad train edit')\n", encoding="utf-8")
            (temp_repo / "script/evaluation/unified_evaluator.py").write_text("print('bad eval edit')\n", encoding="utf-8")
            (temp_repo / "rosetta/train/model_utils.py").write_text("VALUE = 'low priority helper'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "broad implementation"}

    CodePatchAgent(paths.root, config, artifacts, backend=BroadBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))

    assert ideas[0]["code_patch"]["status"] == "ok"
    assert patch["changed_files"] == [
        "rosetta/model/aligner.py",
        "rosetta/model/auto_prune_scope.py",
        "rosetta/model/projector.py",
        "test/test_aligner_span_overlap.py",
    ]
    assert validation["risk_check"]["status"] == "ok"
    prune_actions = [
        action
        for action in patch["recovery_actions"]
        if action["action"] in {
            "auto_prune_worktree_scope_before_build",
            "auto_prune_patch_scope_before_freeze",
            "auto_prune_worktree_and_patch_scope",
        }
    ]
    assert prune_actions
    assert set(prune_actions[0]["restored_files"]) == {
        "script/evaluation/unified_evaluator.py",
        "script/train/SFT_train.py",
        "rosetta/train/model_utils.py",
    }


def test_code_patch_auto_prunes_deleted_files_before_patch_build(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/train").mkdir(parents=True, exist_ok=True)
    (repo / "rosetta/train/answer_prior.py").write_text("VALUE = 'prior'\n", encoding="utf-8")
    (repo / "script/analysis").mkdir(parents=True, exist_ok=True)
    (repo / "script/analysis/route1_eval_flip_diagnostics.py").write_text("VALUE = 'analysis'\n", encoding="utf-8")
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 1
    config["code_patch"]["validation"]["max_contract_repair_attempts"] = 1
    config["code_patch"]["validation"]["max_changed_files"] = 4
    config["code_patch"]["validation"]["auto_prune_over_scope_files"] = True
    paths = init_workspace(config, "topic", project_id="proj_auto_prune_deleted_before_build", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [
        {
            "id": "auto_prune_deleted",
            "title": "Auto Prune Deleted",
            "hypothesis": "h",
            "experiment_contract": {
                "expected_files": [
                    "rosetta/model/aligner.py",
                    "rosetta/model/projector.py",
                    "rosetta/model/auto_prune_deleted.py",
                    "test/test_aligner_span_overlap.py",
                ]
            },
            "implementation_plan": {
                "required_new_files": ["rosetta/model/auto_prune_deleted.py"],
                "smoke_tests": ["test/test_aligner_span_overlap.py"],
            },
        }
    ]

    class DeleteAndBroadBackend:
        def __init__(self):
            self.calls = 0

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            self.calls += 1
            (temp_repo / "rosetta/train/answer_prior.py").unlink()
            (temp_repo / "script/analysis/route1_eval_flip_diagnostics.py").unlink()
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'aligner mechanism'\n", encoding="utf-8")
            (temp_repo / "rosetta/model/projector.py").write_text("VALUE = 'projector mechanism'\n", encoding="utf-8")
            (temp_repo / "rosetta/model/auto_prune_deleted.py").write_text("VALUE = 'new mechanism'\n", encoding="utf-8")
            (temp_repo / "test/test_aligner_span_overlap.py").write_text("def test_scope():\n    assert True\n", encoding="utf-8")
            (temp_repo / "script/evaluation/unified_evaluator.py").write_text("print('bad eval edit')\n", encoding="utf-8")
            return {"status": "ok", "rationale": "deleted unrelated files and touched evaluator"}

    backend = DeleteAndBroadBackend()
    CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))

    assert backend.calls == 1
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert patch["changed_files"] == [
        "rosetta/model/aligner.py",
        "rosetta/model/auto_prune_deleted.py",
        "rosetta/model/projector.py",
        "test/test_aligner_span_overlap.py",
    ]
    assert validation["risk_check"]["status"] == "ok"
    prune_actions = [
        action
        for action in patch["recovery_actions"]
        if action["action"] in {
            "auto_prune_worktree_scope_before_build",
            "auto_prune_worktree_and_patch_scope",
        }
    ]
    assert prune_actions
    restored = set(prune_actions[0]["restored_files"])
    assert "rosetta/train/answer_prior.py" in restored
    assert "script/analysis/route1_eval_flip_diagnostics.py" in restored
    assert "script/evaluation/unified_evaluator.py" in restored


def test_c2c_pipeline_blocks_with_missing_reference_path(monkeypatch, tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "missing_paper.pdf"
    ref_rebuttal = tmp_path / "rebuttal.md"
    ref_rebuttal.write_text("review text", encoding="utf-8")
    config = _base_config(tmp_path / "workspace", simulate=True)
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    orchestrator = Orchestrator()
    project_id = orchestrator.init_c2c_project(
        "cross tokenizer cache",
        target_repo=source_repo,
        ref_paper=ref_paper,
        ref_rebuttal=ref_rebuttal,
        env_python=Path("/usr/bin/python3"),
        project_id="proj_c2c_missing_ref",
        simulate=True,
    )

    result = orchestrator.start(project_id)

    assert result["status"] == "blocked"
    assert result["stage"] == "S0_intake"
    assert "ref_paper" in result["reason"]


def test_c2c_s0_reuses_cached_static_bundle_and_restores_sidecars(monkeypatch, tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(source_repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "allowed_files": ["rosetta/model/aligner.py"],
    }
    shared_memory_path = tmp_path / "shared_method_memory.jsonl"
    config["orchestration"] = {
        "shared_method_memory": {
            "enabled": True,
            "path": str(shared_memory_path),
            "summary_path": str(tmp_path / "shared_method_memory.md"),
        }
    }
    shared_memory_path.write_text(
        json.dumps(
            {
                "schema_version": "shared_method_failure_memory_v1",
                "timestamp": "2026-06-09T00:00:00+00:00",
                "memory_id": "shared-mmlu-regression",
                "project_id": "old_project",
                "kind": "shared_c2c_method_failure",
                "failure_class": "method_failure",
                "route": "proxy_rejected_same_direction",
                "summary": {
                    "failed_idea_ids": ["old_bad_method"],
                    "dataset_regressions": {"mmlu-redux": 2.5},
                    "dragging_datasets": [
                        {"dataset": "mmlu-redux", "sample_family": "multi_domain_knowledge_reasoning", "regression": 2.5}
                    ],
                    "avoid_repeat_rules": ["Do not repeat old_bad_method without fixing mmlu-redux regression."],
                },
                "entries": [
                    {
                        "kind": "c2c_performance_feedback",
                        "idea_id": "old_bad_method",
                        "decision": "proxy_rejected",
                        "proxy_screen": {
                            "proxy_dataset_deltas": {"mmlu-redux": -2.5},
                            "proxy_delta_vs_baseline": -0.8,
                        },
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    paths = init_workspace(config, "topic", project_id="proj_c2c_cached_s0", simulate=False)
    project_root = paths.root
    paper_full = project_root / "intake" / "papers" / "paper_full.md"
    paper_full.parent.mkdir(parents=True, exist_ok=True)
    paper_full.write_text("# Cached Paper\n\nMethod text.\n", encoding="utf-8")
    code_chunk = {
        "chunk_id": "code:rosetta/model/aligner.py::align",
        "path": "rosetta/model/aligner.py",
        "node_type": "function_definition",
        "symbol": "align",
        "start_line": 1,
        "end_line": 3,
        "edit_surface": "allowed",
        "text": "def align():\n    return True\n",
        "text_preview": "def align(): return True",
        "keywords": ["align", "cache"],
    }
    paper_chunk = {
        "chunk_id": "paper:cached:method",
        "source_type": "paper",
        "source_path": "intake/papers/paper_full.md",
        "section": "Method",
        "keywords": ["cache"],
        "text_preview": "Method text.",
        "text": "Method text.",
    }
    rebuttal_chunk = {
        "chunk_id": "rebuttal:cached:review",
        "source_type": "rebuttal",
        "source_path": "intake/papers/rebuttal.md",
        "section": "Review",
        "keywords": ["risk"],
        "text_preview": "Reviewer concern.",
        "text": "Reviewer concern.",
    }
    chunk_index = {
        "schema_version": "c2c_full_chunk_index_v1",
        "counts": {"paper": 1, "rebuttal": 1, "code": 1, "total": 3},
        "entries": [paper_chunk, rebuttal_chunk, {**code_chunk, "source_type": "code", "source_path": "rosetta/model/aligner.py", "section": "align"}],
    }
    bundle = {
        "schema_version": "c2c_static_intake_bundle_v1",
        "metadata": [{"paper_id": "cached", "title": "Cached Paper", "source_path": "paper.pdf", "local_path": "intake/papers/paper_full.md", "kind": "ref_paper"}],
        "reference_result": {"cards": [{"paper_id": "cached", "kind": "ref_paper", "local_path": "intake/papers/paper_full.md"}]},
        "paper_full_manifest": [{"paper_id": "cached", "paper_full_md_path": "intake/papers/paper_full.md", "cache_status": "local_hit"}],
        "repo_manifest": {"files": []},
        "historical_results": {"results": []},
        "baseline": config["c2c"]["baseline"],
        "repo_card": {
            "editable_surface": {"allowed_files": ["rosetta/model/aligner.py"], "allowed_prefixes": []},
            "protocol_constraints": ["Keep protocol fixed."],
        },
        "paper_cards": [{"paper_id": "cached", "title": "Cached Paper", "kind": "ref_paper", "text": "Method text."}],
        "paper_chunks": [paper_chunk],
        "bibliography_cards": [],
        "rebuttal_matrix": {"top_concerns": []},
        "rebuttal_chunks": [rebuttal_chunk],
        "code_cards": [{"path": "rosetta/model/aligner.py", "summary": "Aligner", "symbols": ["align"]}],
        "code_file_manifest": {"files": [{"path": "rosetta/model/aligner.py"}]},
        "code_symbols": [{"symbol_id": "rosetta/model/aligner.py::align", "path": "rosetta/model/aligner.py", "symbol": "align"}],
        "code_chunks": [code_chunk],
        "code_edges": [],
        "code_repo_map": {"counts": {"files": 1, "symbols": 1, "chunks": 1, "edges": 0}, "top_editable_symbols": [{"path": "rosetta/model/aligner.py", "symbol": "align"}]},
        "code_intake_report": {"counts": {"files": 1, "python_files": 1, "symbols": 1, "chunks": 1, "edges": 0, "editable_chunks": 1}, "cache": {"enabled": True, "counts": {"hit": 1}}},
        "implementation_surface_map": {"surfaces": {"rosetta/model/aligner.py": {"symbols": ["align"]}}},
        "code_retrieval_index": {"default_queries": [{"query": "aligner", "chunk_ids": [code_chunk["chunk_id"]]}]},
        "cache_summary": {"schema_version": "c2c_static_cache_summary_v1", "code_intake": {"counts": {"hit": 1}}},
        "chunk_index": chunk_index,
        "result_ledger_csv": "id,mean\n",
        "negative_memory": {"blocked_idea_patterns": []},
        "retrieval_plan": {
            "questions": [
                {
                    "question_id": "mechanism",
                    "question": "Which code paths implement cache routing?",
                    "priority_terms": ["cache", "aligner"],
                }
            ]
        },
        "followup_bundle": {
            "questions": [
                {
                    "question_id": "mechanism",
                    "cross_source_targets": [{"source_type": "code", "source_path": "rosetta/model/aligner.py"}],
                }
            ]
        },
        "evidence_brief": {"schema_version": "c2c_evidence_brief_v1", "topic": "topic"},
    }
    bundle["validity"] = _c2c_static_bundle_validity(project_root, config)
    write_json(project_root / "intake" / "c2c" / "static_bundle.json", bundle)

    def fail_import(self):
        raise AssertionError("S0 should reuse cached static_bundle instead of re-importing references")

    monkeypatch.setattr(C2CAdapter, "import_reference_materials", fail_import)
    context = AgentContext(project_root, config, ArtifactManager(project_root), ModelClient(config, project_root=project_root))

    result = IntakeAgent(context).run("topic")

    assert result["status"] == "ok"
    assert result["cache_status"] == "reused"
    assert (project_root / "intake/c2c/chunk_index.jsonl").exists()
    assert (project_root / "intake/c2c/code_chunks.jsonl").exists()
    assert (project_root / "intake/c2c/code_repo_map.md").exists()
    assert (project_root / "intake/c2c/code_intake_report.md").exists()
    shared_snapshot = json.loads((project_root / "intake/shared_method_failure_memory.json").read_text(encoding="utf-8"))
    negative_memory = json.loads((project_root / "intake/c2c/negative_result_memory.json").read_text(encoding="utf-8"))
    evidence_brief = json.loads((project_root / "intake/c2c/evidence_brief.json").read_text(encoding="utf-8"))
    refreshed_bundle = json.loads((project_root / "intake/c2c/static_bundle.json").read_text(encoding="utf-8"))
    assert shared_snapshot["entry_count"] == 1
    assert "old_bad_method" in shared_snapshot["entries"][0]["summary"]["failed_idea_ids"]
    assert shared_snapshot["retrieval_policy"]["mode"] == "quality_weighted_top_k_retrieval"
    assert shared_snapshot["retrieval_context"]["datasets"] == ["mmlu-redux"]
    assert shared_snapshot["ranking_policy"]["sort"] == "descending memory_quality.priority"
    assert "shared-mmlu-regression" in shared_snapshot["quality_summary"]["high_quality_memory_ids"]
    assert negative_memory["shared_method_memory"]["entry_count"] == 1
    assert "shared-mmlu-regression" in negative_memory["shared_method_memory"]["high_quality_memory_ids"]
    assert negative_memory["shared_method_memory"]["memory_catalog"][0]["memory_id"] == "shared-mmlu-regression"
    assert "read_hint" in negative_memory["shared_method_memory"]["memory_catalog"][0]
    assert "dataset_regression" in negative_memory["shared_method_memory"]["quality_summary"]["signal_counts"]
    assert "shared-mmlu-regression" in evidence_brief["shared_method_memory"]["high_quality_memory_ids"]
    assert evidence_brief["shared_method_memory"]["memory_catalog"][0]["memory_id"] == "shared-mmlu-regression"
    assert refreshed_bundle["shared_method_memory"]["entry_count"] == 1
    assert refreshed_bundle["negative_memory"]["shared_method_memory"]["entry_count"] == 1
    assert refreshed_bundle["evidence_brief"]["shared_method_memory"]["entry_count"] == 1
    assert refreshed_bundle["validity"]["fingerprint"] == _c2c_static_bundle_validity(project_root, config)["fingerprint"]
    assert "Do not repeat old_bad_method" in negative_memory["blocked_idea_patterns"][0]
    manifest = json.loads((project_root / "intake/stage_manifest.json").read_text(encoding="utf-8"))
    artifact_paths = {item["path"] for item in manifest["artifacts"]}
    assert "intake/c2c/static_bundle.json" in artifact_paths
    assert "intake/c2c/chunk_index.jsonl" in artifact_paths
    assert gate_s0(project_root, config).passed is True


def test_c2c_s0_rejects_stale_cached_static_bundle(tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(source_repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "allowed_files": ["rosetta/model/aligner.py"],
    }
    paths = init_workspace(config, "topic", project_id="proj_c2c_stale_s0", simulate=False)
    project_root = paths.root
    code_chunk = {
        "chunk_id": "code:align",
        "source_type": "code",
        "source_path": "rosetta/model/aligner.py",
        "path": "rosetta/model/aligner.py",
        "node_type": "function_definition",
        "symbol": "align",
        "start_line": 1,
        "end_line": 2,
        "edit_surface": "allowed",
        "text": "def align(): pass\n",
        "keywords": ["align"],
        "section": "align",
        "text_preview": "def align(): pass",
    }
    paper_chunk = {"chunk_id": "paper:1", "source_type": "paper", "source_path": "paper.md", "section": "Method", "keywords": ["cache"], "text_preview": "paper", "text": "paper"}
    rebuttal_chunk = {"chunk_id": "rebuttal:1", "source_type": "rebuttal", "source_path": "rebuttal.md", "section": "Review", "keywords": ["risk"], "text_preview": "risk", "text": "risk"}
    bundle = {
        "schema_version": "c2c_static_intake_bundle_v1",
        "paper_chunks": [paper_chunk],
        "rebuttal_chunks": [rebuttal_chunk],
        "code_file_manifest": {"files": [{"path": "rosetta/model/aligner.py"}]},
        "code_symbols": [{"path": "rosetta/model/aligner.py", "symbol": "align"}],
        "code_chunks": [code_chunk],
        "code_edges": [],
        "code_repo_map": {"counts": {"chunks": 1}},
        "code_intake_report": {"counts": {"chunks": 1}},
        "implementation_surface_map": {"surfaces": {"rosetta/model/aligner.py": {}}},
        "code_retrieval_index": {"default_queries": [{"query": "align"}]},
        "cache_summary": {},
        "paper_full_manifest": [],
        "evidence_brief": {"schema_version": "c2c_evidence_brief_v1"},
        "baseline": config["c2c"]["baseline"],
        "repo_card": {"editable_surface": {"allowed_files": ["rosetta/model/aligner.py"], "allowed_prefixes": []}},
        "paper_cards": [{"paper_id": "p1", "title": "Paper", "kind": "ref_paper", "text": "paper"}],
        "rebuttal_matrix": {"top_concerns": []},
        "code_cards": [{"path": "rosetta/model/aligner.py", "summary": "Aligner"}],
        "negative_memory": {"blocked_idea_patterns": []},
        "retrieval_plan": {"questions": [{"question_id": "mechanism", "question": "code?", "priority_terms": ["align"]}]},
        "followup_bundle": {"questions": [{"question_id": "mechanism", "cross_source_targets": [{"source_type": "code", "source_path": "rosetta/model/aligner.py"}]}]},
        "chunk_index": {"counts": {"paper": 1, "rebuttal": 1, "code": 1, "total": 3}, "entries": [paper_chunk, rebuttal_chunk, code_chunk]},
        "validity": _c2c_static_bundle_validity(project_root, config),
    }
    write_json(project_root / "intake" / "c2c" / "static_bundle.json", bundle)

    context = AgentContext(project_root, config, ArtifactManager(project_root), ModelClient(config, project_root=project_root))
    assert IntakeAgent(context)._load_reusable_c2c_static_bundle(_c2c_static_bundle_validity(project_root, config)) is not None

    (source_repo / "rosetta/model/aligner.py").write_text("VALUE = 'changed'\n", encoding="utf-8")

    assert IntakeAgent(context)._load_reusable_c2c_static_bundle(_c2c_static_bundle_validity(project_root, config)) is None


def test_bootstrap_cached_s0_only_blocks_before_external_reference_parsing(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    paper = tmp_path / "paper.md"
    rebuttal = tmp_path / "rebuttal.md"
    paper.write_text("paper", encoding="utf-8")
    rebuttal.write_text("rebuttal", encoding="utf-8")
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "target_repo": str(repo),
        "snapshot_path": str(repo),
        "ref_paper": str(paper),
        "ref_rebuttal": str(rebuttal),
        "env_python": sys.executable,
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "allowed_files": ["rosetta/model/aligner.py"],
        "allowed_prefixes": ["recipe/"],
    }
    config["orchestration"] = {
        "profile": "bootstrap",
        "bootstrap": {"cached_s0_only": True},
    }
    paths = init_workspace(config, "topic", project_id="proj_cached_s0_required", simulate=False)

    def fail_import(self):
        raise AssertionError("cached-S0-only bootstrap must not call MinerU/reference parsing")

    monkeypatch.setattr(C2CAdapter, "import_reference_materials", fail_import)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))

    result = IntakeAgent(context).run("topic")

    assert result["status"] == "blocked"
    assert "DeepSeek and MinerU fallback are disabled" in result["blocked_reason"]
    assert (paths.root / "intake/c2c/cache_required_blocked.json").exists()


def test_c2c_evidence_brief_uses_current_repo_and_retrieval_fields() -> None:
    brief = _c2c_evidence_brief(
        topic="cache routing",
        baseline={"name": "base", "mean": 50.0},
        repo_card={
            "editable_surface": {"allowed_files": ["rosetta/model/aligner.py"], "allowed_prefixes": ["rosetta/model"]},
            "protocol_constraints": ["Keep receiver/sharer fixed."],
        },
        paper_cards=[{"paper_id": "p1", "title": "Paper", "kind": "ref_paper", "text": "Method text."}],
        rebuttal_matrix={"top_concerns": ["baseline fairness"], "structured_concerns": [{"concern": "baseline"}]},
        code_cards=[{"path": "rosetta/model/aligner.py", "summary": "Aligner", "symbols": ["align"]}],
        negative_memory={"blocked_idea_patterns": ["hard gate"]},
        retrieval_plan={
            "questions": [
                {
                    "question_id": "mechanism",
                    "question": "Which code paths implement cache routing?",
                    "priority_terms": ["cache", "aligner"],
                }
            ]
        },
        followup_bundle={
            "questions": [
                {
                    "question_id": "mechanism",
                    "cross_source_targets": [
                        {"source_type": "paper", "source_path": "paper.md"},
                        {"source_type": "code", "source_path": "rosetta/model/aligner.py"},
                    ],
                }
            ]
        },
    )

    assert brief["repo_summary"]["editable_surface"]["allowed_files"] == ["rosetta/model/aligner.py"]
    assert brief["repo_summary"]["protocol_constraints"] == ["Keep receiver/sharer fixed."]
    assert brief["retrieval_targets"]["questions"][0]["question_id"] == "mechanism"
    assert {item["source_type"] for item in brief["retrieval_targets"]["cross_source_targets"]} == {"paper", "code"}


def test_c2c_s0_semantic_enrichment_missing_key_fails_open(monkeypatch, tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "paper.txt"
    ref_rebuttal = tmp_path / "rebuttal.md"
    ref_paper.write_text("paper method text", encoding="utf-8")
    ref_rebuttal.write_text("review concern text", encoding="utf-8")
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["intake"] = {"semantic_enrichment": {"enabled": True, "provider": "deepseek", "model": "deepseek-v4-flash"}}
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(source_repo),
        "ref_paper": str(ref_paper),
        "ref_rebuttal": str(ref_rebuttal),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "allowed_files": ["rosetta/model/aligner.py"],
    }
    paths = init_workspace(config, "topic", project_id="proj_c2c_s0_enrich_fail_open", simulate=False)

    def fail_enrichment(self, **kwargs):
        raise S0SemanticEnrichmentError("DEEPSEEK_API_KEY is missing")

    monkeypatch.setattr(DeepSeekS0SemanticEnricher, "enrich_c2c_chunks", fail_enrichment)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))

    result = IntakeAgent(context).run("topic")

    assert result["status"] == "ok"
    bundle = json.loads((paths.root / "intake/c2c/static_bundle.json").read_text(encoding="utf-8"))
    assert bundle["semantic_enrichment"]["status"] == "failed_open"
    assert bundle["semantic_enrichment"]["fallback"] == "raw_chunks_without_semantic_enrichment"
    assert bundle["chunk_index"]["counts"]["paper"] > 0
    assert bundle["chunk_index"]["counts"]["code"] > 0
    assert gate_s0(paths.root, config).passed is True


def test_c2c_s0_semantic_enrichment_rebuilds_code_indexes(monkeypatch, tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "paper.txt"
    ref_rebuttal = tmp_path / "rebuttal.md"
    ref_paper.write_text("paper method text", encoding="utf-8")
    ref_rebuttal.write_text("review concern text", encoding="utf-8")
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["intake"] = {"semantic_enrichment": {"enabled": True, "provider": "deepseek", "model": "deepseek-v4-flash"}}
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(source_repo),
        "ref_paper": str(ref_paper),
        "ref_rebuttal": str(ref_rebuttal),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "allowed_files": ["rosetta/model/aligner.py"],
    }
    paths = init_workspace(config, "topic", project_id="proj_c2c_s0_enrich_indexes", simulate=False)

    def enrich_with_code_semantics(self, **kwargs):
        code_chunks = []
        for chunk in kwargs["code_chunks"]:
            item = dict(chunk)
            if item.get("path") == "rosetta/model/aligner.py":
                item["semantic_summary"] = "Aligner handles valid_mask routing for cache transfer."
                item["mechanism_tags"] = ["alignment_core", "cache_routing"]
                item["retrieval_keywords"] = ["alignment", "valid_mask", "routing"]
                item["semantic_enrichment"] = {"schema_version": "s0_semantic_enrichment_sample_v1", "cache_status": "test"}
            code_chunks.append(item)
        return {
            "paper_chunks": kwargs["paper_chunks"],
            "rebuttal_chunks": kwargs["rebuttal_chunks"],
            "code_chunks": code_chunks,
            "report": {"enabled": True, "status": "ok", "records": []},
            "artifacts": [],
        }

    monkeypatch.setattr(DeepSeekS0SemanticEnricher, "enrich_c2c_chunks", enrich_with_code_semantics)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))

    result = IntakeAgent(context).run("topic")

    assert result["status"] == "ok"
    bundle = json.loads((paths.root / "intake/c2c/static_bundle.json").read_text(encoding="utf-8"))
    aligner_chunk = next(chunk for chunk in bundle["code_chunks"] if chunk.get("path") == "rosetta/model/aligner.py")
    assert aligner_chunk["retrieval_keywords"] == ["alignment", "valid_mask", "routing"]
    alignment_surface = bundle["implementation_surface_map"]["surfaces"]["alignment_core"]
    assert any(item.get("semantic_summary") == "Aligner handles valid_mask routing for cache transfer." for item in alignment_surface)
    alignment_query = bundle["code_retrieval_index"]["default_queries"][0]
    aligner_result = next(item for item in alignment_query["results"] if item.get("path") == "rosetta/model/aligner.py")
    assert "retrieval_keywords:valid_mask" in aligner_result["match_reasons"]
    assert aligner_result["semantic_summary"] == "Aligner handles valid_mask routing for cache transfer."
    assert gate_s0(paths.root, config).passed is True


def test_c2c_pipeline_runs_to_s3_with_mock_small_loop(monkeypatch, tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "paper.txt"
    ref_rebuttal = tmp_path / "rebuttal.md"
    ref_paper.write_text(
        (
            "Method evidence. Cache transfer needs a utility signal for routing and mechanism selection. "
            "The baseline transfers hidden states without downstream utility prediction, so a learned "
            "soft routing signal can preserve useful spans while reducing harmful transfer. "
        )
        * 18
        + "\n\n"
        + (
            "Coverage evidence. Coverage-preserving transfer avoids regressions by keeping the original "
            "communication path available and using diagnostics for span coverage, dataset regressions, "
            "and ablation controls. "
        )
        * 18,
        encoding="utf-8",
    )
    ref_rebuttal.write_text("reviewer risk failure coverage regression counterevidence", encoding="utf-8")
    config = _base_config(tmp_path / "workspace", simulate=True)
    config["llm"].update({"model": "gpt-5.6-terra", "reasoning_effort": "xhigh"})
    config["agents"] = {"s2_directional_planner": {"resume_enabled": False}}
    config.setdefault("orchestration", {})["shared_method_memory"] = {
        "enabled": True,
        "path": str(tmp_path / "s1_shared_memory.jsonl"),
        "summary_path": str(tmp_path / "s1_shared_memory.md"),
    }
    shared_memory_entry = {
        "schema_version": "shared_method_failure_memory_v1",
        "memory_id": "mem_s1_avoid_hard_gate",
        "timestamp": "2026-06-10T00:00:00Z",
        "project_id": "old_project",
        "topic": "cross tokenizer cache",
        "kind": "shared_c2c_method_failure",
        "route": "proxy_rejected_same_direction",
        "failure_class": "method_failure",
        "summary": {
            "failed_idea_ids": ["hard_gate_stack"],
            "dragging_datasets": [{"dataset": "mmlu-redux", "regression": 2.0}],
            "avoid_repeat_rules": ["Avoid extra hard accept/reject gates that collapse transfer coverage."],
        },
        "entries": [{"id": "hard_gate_stack", "decision": "proxy_rejected"}],
    }
    Path(config["orchestration"]["shared_method_memory"]["path"]).write_text(json.dumps(shared_memory_entry, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    monkeypatch.setattr(literature_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    s1_codex_commands = []
    s1_codex_prompts = []
    original_subprocess_run = literature_module.subprocess.run

    def prompt_context(prompt: str) -> dict:
        return json.loads(prompt.split("Context JSON:", 1)[1].split("Required JSON shape:", 1)[0])

    def direction_from_allowed_refs(prompt: str) -> dict:
        context = prompt_context(prompt)
        allowed_refs = [ref for ref in context["allowed_refs"] if isinstance(ref, dict)]
        paper_refs = [ref for ref in allowed_refs if ref.get("source_type") == "paper"][:2]
        code_refs = [ref for ref in allowed_refs if ref.get("source_type") == "code"][:2]
        counter_refs = [ref for ref in allowed_refs if ref.get("source_type") in {"rebuttal", "failure_feedback"}][:1]
        expected_files = [ref.get("source_path") or ref.get("source_label") for ref in code_refs]
        return {
            "schema_version": "c2c_s1_direction_agent_v1",
            "status": "ok",
            "used_shared_memory_refs": ["mem_s1_avoid_hard_gate"],
            "direction_decision": {
                "direction_id": "utility_predicted_cache_routing",
                "mechanism_direction": "Utility Predicted Cache Routing",
                "mechanism_type": "utility_predicted_cache_routing",
                "mechanism_axis": "routing",
                "integration_point": "wrapper",
                "control_signal": "utility",
                "core_hypothesis": "Predict downstream utility for transferred cache states and let S2 explore soft routing mechanisms that preserve baseline coverage.",
                "why_baseline_fails": "The baseline lacks downstream utility control.",
                "why_this_direction": "The deterministic bundle contains paper support, code surfaces, and counterevidence against hard gates.",
                "expected_metric_signature": {"primary_metric": "three_dataset_mean", "expected_direction": "increase", "diagnostics": ["transfer coverage"]},
                "required_evidence_refs": paper_refs,
                "counterevidence_refs": counter_refs,
                "implementation_surface_refs": code_refs,
                "expected_files": expected_files,
                "allowed_variants": ["soft residual utility scaling", "coverage-preserving utility modulation"],
                "forbidden_patterns": ["extra hard accept/reject gate", "evaluator changes"],
                "failure_routing_hints": ["return to S1 if coverage-preserving routing repeatedly collapses"],
                "s2_affordance": "S2 can instantiate utility-modulated routing on the retrieved code surfaces.",
                "target_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
                "failure_focus": ["dataset-level coverage collapse", "mmlu-redux regression"],
                "verification_commands": ["py_compile", "small2048_train", "three_dataset_eval"],
                "used_shared_memory_refs": ["mem_s1_avoid_hard_gate"],
            },
            "selected_ideas": [
                {
                    "id": "utility_predicted_cache_routing",
                    "title": "Utility Predicted Cache Routing",
                    "selected": True,
                    "hypothesis": "Predict downstream utility for transferred cache states and route them without reducing baseline transfer coverage.",
                    "novelty_score": 7,
                    "feasibility_score": 7,
                    "mechanism_type": "utility_predicted_cache_routing",
                    "description": "High-level S1 direction only; S2 will generate concrete implementation candidates.",
                    "motivation": "Baseline transfer lacks a downstream utility signal and previous failures warn against hard gating.",
                    "reviewer_risk_response": "Track transfer coverage and per-dataset regressions; forbid evaluator edits and hard-gate stacking.",
                    "expected_files": expected_files,
                    "verification_commands": ["py_compile", "small2048_train", "three_dataset_eval"],
                    "evidence_refs": paper_refs,
                    "counterevidence_refs": counter_refs,
                    "code_refs": code_refs,
                    "s1_allowed_variants": ["soft residual utility scaling", "coverage-preserving utility modulation"],
                    "s1_forbidden_patterns": ["extra hard accept/reject gate", "evaluator changes"],
                    "used_shared_memory_refs": ["mem_s1_avoid_hard_gate"],
                }
            ],
            "candidate_direction_scorecard": {
                "schema_version": "c2c_s1_direction_candidate_scorecard_v1",
                "selected_direction_id": "utility_predicted_cache_routing",
                "candidates": [
                    {
                        "direction_id": "utility_predicted_cache_routing",
                        "mechanism_axis": "routing",
                        "integration_point": "wrapper",
                        "control_signal": "utility",
                        "score": 0.86,
                        "selected": True,
                        "evidence_refs": paper_refs,
                        "counterevidence_refs": counter_refs,
                        "implementation_surface_refs": code_refs,
                        "why_selected": "It has paper support, editable wrapper/code refs, and known hard-gate counterevidence.",
                        "why_not_selected": [],
                    },
                    {
                        "direction_id": "alignment_surface_signal",
                        "mechanism_axis": "alignment",
                        "integration_point": "aligner",
                        "control_signal": "representation_match",
                        "score": 0.51,
                        "selected": False,
                        "evidence_refs": paper_refs[:1],
                        "counterevidence_refs": counter_refs,
                        "implementation_surface_refs": code_refs[:1],
                        "why_selected": "",
                        "why_not_selected": ["Less direct evidence that alignment alone explains the baseline failure."],
                    },
                ],
                "comparison_axes": ["evidence_support", "counterevidence_resolution", "implementation_surface"],
            },
            "negative_constraints": {
                "reviewer_concerns": ["failure_modes_ood", "coverage collapse"],
                "forbidden_idea_ids": ["hard_gate_stack"],
                "forbidden_patterns": ["extra hard accept/reject gate", "evaluator changes"],
                "failure_feedback_rules": ["Use method-level failures only; ignore S2.5 coding noise in S1."],
                "used_shared_memory_refs": ["mem_s1_avoid_hard_gate"],
            },
            "decision_chain": {
                "evidence": ["paper_support", "code_surface"],
                "counterevidence": ["counterevidence"],
                "conclusion": "Use utility-predicted cache routing as the S1 direction and let S2 choose concrete variants.",
            },
        }

    def fake_s1_codex_run(command, **kwargs):
        if not command or command[0] != "codex":
            return original_subprocess_run(command, **kwargs)
        s1_codex_commands.append(command)
        prompt = kwargs.get("input") or ""
        s1_codex_prompts.append(prompt)
        output_path = Path(command[command.index("--output-last-message") + 1])
        if len(s1_codex_commands) == 1:
            output_path.write_text("not json", encoding="utf-8")
            stdout = '{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614174001"}\n'
        elif "evidence_request_agent" in prompt:
            output_path.write_text(json.dumps(literature_module.default_c2c_evidence_request_plan(topic="cross tokenizer cache")), encoding="utf-8")
            stdout = ""
        else:
            output_path.write_text(json.dumps(direction_from_allowed_refs(prompt)), encoding="utf-8")
            stdout = ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(literature_module.subprocess, "run", fake_s1_codex_run)
    orchestrator = Orchestrator()
    project_id = orchestrator.init_c2c_project(
        "cross tokenizer cache",
        target_repo=source_repo,
        ref_paper=ref_paper,
        ref_rebuttal=ref_rebuttal,
        env_python=Path("/usr/bin/python3"),
        project_id="proj_c2c_loop",
        simulate=True,
    )
    result = orchestrator.start(project_id)
    root = tmp_path / "workspace" / project_id

    assert result["status"] == "completed"
    main_results = json.loads((root / "experiment/results/main_results.json").read_text(encoding="utf-8"))
    assert main_results["best_candidate"]["decision"] == "candidate_win"
    assert main_results["acceptance"]["passed"] is True
    assert main_results["acceptance"]["delta"] >= main_results["acceptance"]["min_delta_to_pass"]
    assert (root / "intake/c2c/static_bundle.json").exists()
    assert (root / "intake/c2c/evidence_brief.json").exists()
    assert (root / "intake/c2c/chunk_index.json").exists()
    assert (root / "intake/c2c/chunk_index.jsonl").exists()
    assert (root / "intake/c2c/code_intake_report.json").exists()
    assert (root / "intake/c2c/implementation_surface_map.json").exists()
    assert (root / "intake/c2c/code_retrieval_index.json").exists()
    assert (root / "plan/short_loop_plan.yaml").exists()
    assert (root / "literature/idea_debate.json").exists()
    assert (root / "literature/negative_constraints.json").exists()
    assert (root / "literature/direction.json").exists()
    assert (root / "literature/direction_scorecard.json").exists()
    assert (root / "literature/evidence_bundle.json").exists()
    assert (root / "literature/novelty_audit.json").exists()
    assert (root / "literature/c2c/evidence_request_plan.json").exists()
    assert (root / "literature/c2c/evidence_requests.json").exists()
    assert (root / "literature/c2c/evidence_bundle.json").exists()
    assert (root / "literature/c2c/direction_decision.json").exists()
    assert (root / "literature/c2c/direction_candidate_scorecard.json").exists()
    assert (root / "literature/c2c/evidence_session.json").exists()
    assert (root / "literature/c2c/evidence_quality_score.json").exists()
    assert (root / "literature/c2c/evidence_retrieval_trace.json").exists()
    assert (root / "literature/c2c/direction_fingerprint.json").exists()
    assert (root / "literature/c2c/repo_card.json").exists()
    assert (root / "literature/c2c/rebuttal_concern_matrix.json").exists()
    assert (root / "literature/c2c/negative_result_memory.json").exists()
    assert (root / "literature/c2c/paper_chunks.jsonl").exists()
    assert (root / "literature/c2c/bibliography.json").exists()
    assert (root / "literature/c2c/rebuttal_chunks.jsonl").exists()
    assert (root / "literature/c2c/code_cards.json").exists()
    assert (root / "literature/c2c/code_chunks.jsonl").exists()
    assert (root / "literature/c2c/code_intake_report.json").exists()
    assert (root / "literature/c2c/implementation_surface_map.json").exists()
    assert (root / "literature/c2c/code_retrieval_index.json").exists()
    assert (root / "literature/c2c/chunk_index.json").exists()
    assert (root / "literature/c2c/retrieval_plan.json").exists()
    assert (root / "literature/c2c/retrieval_followup.json").exists()
    assert (root / "plan/planner_decision.json").exists()
    assert (root / "plan/variant_contract.json").exists()
    assert (root / "plan/variant_fingerprint.json").exists()
    bundle = json.loads((root / "intake/c2c/static_bundle.json").read_text(encoding="utf-8"))
    assert bundle["chunk_index"]["counts"]["paper"] > 0
    assert bundle["chunk_index"]["counts"]["rebuttal"] > 0
    assert bundle["chunk_index"]["counts"]["code"] > 0
    assert bundle["code_intake_report"]["counts"]["chunks"] > 0
    assert bundle["implementation_surface_map"]["surfaces"]
    assert bundle["code_retrieval_index"]["default_queries"]
    evidence_quality = json.loads((root / "literature/c2c/evidence_quality_score.json").read_text(encoding="utf-8"))
    assert evidence_quality["gate"] == "pass"
    assert evidence_quality["support_coverage"]["paper"] >= 2
    assert evidence_quality["support_coverage"]["code"] >= 2
    retrieval_trace = json.loads((root / "literature/c2c/evidence_retrieval_trace.json").read_text(encoding="utf-8"))
    assert retrieval_trace["deterministic"] is True
    assert retrieval_trace["schema_version"] == "c2c_s1_deterministic_retrieval_trace_v1"
    deterministic_bundle = json.loads((root / "literature/c2c/evidence_bundle.json").read_text(encoding="utf-8"))
    assert deterministic_bundle["producer"] == "deterministic_retriever"
    ideas = json.loads((root / "literature/ideas.json").read_text(encoding="utf-8"))
    assert len(ideas) == 1
    assert ideas[0]["id"] == "utility_predicted_cache_routing"
    assert ideas[0]["used_shared_memory_refs"] == ["mem_s1_avoid_hard_gate"]
    assert ideas[0]["s1_evidence_agent"]["source"] == "codex_two_phase_direction_agent"
    assert ideas[0]["s1_evidence_agent"]["used_shared_memory_refs"] == ["mem_s1_avoid_hard_gate"]
    direction = json.loads((root / "literature/c2c/direction_decision.json").read_text(encoding="utf-8"))
    assert direction["direction_id"] == "utility_predicted_cache_routing"
    assert direction["used_shared_memory_refs"] == ["mem_s1_avoid_hard_gate"]
    direction_candidate_scorecard = json.loads((root / "literature/c2c/direction_candidate_scorecard.json").read_text(encoding="utf-8"))
    assert direction_candidate_scorecard["selected_direction_id"] == "utility_predicted_cache_routing"
    assert any(candidate["why_not_selected"] for candidate in direction_candidate_scorecard["candidates"] if not candidate["selected"])
    root_direction = json.loads((root / "literature/direction.json").read_text(encoding="utf-8"))
    assert root_direction["direction_id"] == "utility_predicted_cache_routing"
    assert root_direction["mechanism_axis"]
    variant_contract = json.loads((root / "plan/variant_contract.json").read_text(encoding="utf-8"))
    assert variant_contract["direction_id"] == "utility_predicted_cache_routing"
    assert variant_contract["ablation"]["switch"]
    debate = json.loads((root / "literature/idea_debate.json").read_text(encoding="utf-8"))
    assert debate["used_shared_memory_refs"] == ["mem_s1_avoid_hard_gate"]
    constraints = json.loads((root / "literature/negative_constraints.json").read_text(encoding="utf-8"))
    assert constraints["used_shared_memory_refs"] == ["mem_s1_avoid_hard_gate"]
    evidence_session = json.loads((root / "literature/c2c/evidence_session.json").read_text(encoding="utf-8"))
    assert evidence_session["schema_version"] == "c2c_s1_two_phase_session_v1"
    assert evidence_session["repair_count"] == 1
    assert evidence_session["used_shared_memory_refs"] == ["mem_s1_avoid_hard_gate"]
    assert len(evidence_session["attempts"]) == 3
    assert "resume" in s1_codex_commands[1]
    assert "resume" in s1_codex_commands[2]
    assert s1_codex_commands[0][s1_codex_commands[0].index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="xhigh"' in s1_codex_commands[0]
    assert "Validation errors" in s1_codex_prompts[1]


def test_c2c_s1_direction_agent_can_request_followup_cards_in_same_session(monkeypatch, tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=True)
    config["ideation"] = {
        "c2c_s1_two_phase": {"max_direction_followup_rounds": 1},
        "c2c": {"novelty_auditor": {"enabled": False}},
    }
    project_root = tmp_path / "project"
    project_root.mkdir()
    context = AgentContext(project_root, config, ArtifactManager(project_root), ModelClient(config, project_root=project_root))
    agent = literature_module.LiteratureAgent(context)
    monkeypatch.setattr(literature_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    original_subprocess_run = literature_module.subprocess.run
    commands = []
    direction_calls = 0

    def prompt_context(prompt: str) -> dict:
        return json.loads(prompt.split("Context JSON:", 1)[1].split("Required JSON shape:", 1)[0])

    def final_direction(prompt: str) -> dict:
        context_payload = prompt_context(prompt)
        refs = [ref for ref in context_payload["allowed_refs"] if isinstance(ref, dict)]
        paper_refs = [ref for ref in refs if ref.get("source_type") == "paper"][:2]
        code_refs = [ref for ref in refs if ref.get("source_type") == "code"][:2]
        counter_refs = [ref for ref in refs if ref.get("source_type") == "rebuttal"][:1]
        return {
            "schema_version": "c2c_s1_direction_agent_v1",
            "status": "ok",
            "direction_decision": {
                "direction_id": "followup_wrapper_routing",
                "mechanism_direction": "Follow-up wrapper routing",
                "mechanism_type": "routing",
                "mechanism_axis": "routing",
                "integration_point": "wrapper",
                "control_signal": "utility",
                "core_hypothesis": "Use utility-aware wrapper routing after inspecting both initial and follow-up code cards.",
                "why_baseline_fails": "The baseline lacks a utility control signal.",
                "why_this_direction": "The merged deterministic bundle contains paper, counterevidence, and two code surfaces.",
                "expected_metric_signature": {"primary_metric": "three_dataset_mean", "expected_direction": "increase", "diagnostics": []},
                "required_evidence_refs": paper_refs,
                "counterevidence_refs": counter_refs,
                "implementation_surface_refs": code_refs,
                "expected_files": [ref["source_path"] for ref in code_refs],
                "allowed_variants": ["wrapper utility routing"],
                "forbidden_patterns": ["hard gate"],
                "failure_routing_hints": ["return to S1 after repeated method failures"],
                "s2_affordance": "S2 can instantiate wrapper utility routing.",
                "verification_commands": ["py_compile"],
                "target_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
                "failure_focus": ["coverage"],
            },
            "selected_ideas": [
                {
                    "id": "followup_wrapper_routing",
                    "title": "Follow-up wrapper routing",
                    "selected": True,
                    "hypothesis": "Use utility-aware wrapper routing.",
                    "novelty_score": 7,
                    "feasibility_score": 7,
                    "mechanism_type": "routing",
                    "description": "High-level direction selected after follow-up evidence.",
                    "motivation": "Merged deterministic evidence supports wrapper routing.",
                    "reviewer_risk_response": "Track coverage and avoid hard gates.",
                    "expected_files": [ref["source_path"] for ref in code_refs],
                    "verification_commands": ["py_compile"],
                    "evidence_refs": paper_refs,
                    "counterevidence_refs": counter_refs,
                    "code_refs": code_refs,
                    "s1_allowed_variants": ["wrapper utility routing"],
                    "s1_forbidden_patterns": ["hard gate"],
                }
            ],
            "negative_constraints": {"forbidden_patterns": ["hard gate"]},
            "decision_chain": {"evidence": ["paper", "code"], "counterevidence": ["rebuttal"], "conclusion": "select wrapper routing"},
        }

    def fake_codex_run(command, **kwargs):
        nonlocal direction_calls
        if not command or command[0] != "codex":
            return original_subprocess_run(command, **kwargs)
        commands.append(command)
        prompt = kwargs.get("input") or ""
        output_path = Path(command[command.index("--output-last-message") + 1])
        if "evidence_request_agent" in prompt:
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": "c2c_s1_evidence_request_plan_v1",
                        "request_plan_id": "initial_plan",
                        "evidence_requests": [
                            {"request_id": "paper_support", "source_type": "paper", "query": "cache transfer support", "keywords": ["cache", "transfer"], "purpose": "support", "top_k": 2, "must_resolve": True},
                            {"request_id": "code_aligner", "source_type": "code", "query": "aligner wrapper cache", "keywords": ["aligner", "wrapper"], "purpose": "implementation_surface", "top_k": 2, "must_resolve": True},
                            {"request_id": "counter", "source_type": "rebuttal", "query": "coverage risk", "keywords": ["coverage", "risk"], "purpose": "counterevidence", "top_k": 1, "must_resolve": True},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout='{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614174099"}\n', stderr="")
        direction_calls += 1
        if direction_calls == 1:
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": "c2c_s1_direction_agent_v1",
                        "status": "needs_more_evidence",
                        "reason": "Need wrapper implementation surface before choosing.",
                        "followup_evidence_request_plan": {
                            "evidence_requests": [
                                {"request_id": "code_wrapper_followup", "source_type": "code", "query": "wrapper forward utility routing", "keywords": ["wrapper", "forward", "utility"], "purpose": "implementation_surface", "top_k": 1, "must_resolve": True}
                            ],
                            "request_rationale": "The first bundle only covered aligner code.",
                        },
                    }
                ),
                encoding="utf-8",
            )
        else:
            output_path.write_text(json.dumps(final_direction(prompt)), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(literature_module.subprocess, "run", fake_codex_run)
    result = agent._run_c2c_evidence_on_demand_direction(
        topic="cross tokenizer cache",
        evidence_brief={},
        chunk_index={},
        paper_chunks=[
            {"chunk_id": "paper_1", "text": "cache transfer support mechanism", "source_path": "paper.md"},
            {"chunk_id": "paper_2", "text": "transfer utility support evidence", "source_path": "paper.md"},
        ],
        rebuttal_chunks=[{"chunk_id": "reb_1", "text": "coverage risk and regression counterevidence", "source_path": "rebuttal.md"}],
        code_chunks=[
            {"chunk_id": "code_aligner", "text": "aligner cache bridge", "source_path": "rosetta/model/aligner.py"},
            {"chunk_id": "code_wrapper", "text": "wrapper forward utility routing", "source_path": "rosetta/model/wrapper.py"},
        ],
        code_edges=[],
        code_intake_report={},
        implementation_surface_map={"surfaces": {"rosetta/model/aligner.py": {"status": "allowed"}, "rosetta/model/wrapper.py": {"status": "allowed"}}},
        code_retrieval_index={},
        baseline={},
        negative_memory={},
        rebuttal_matrix={},
        feedback=[],
    )

    assert result["status"] == "ok"
    assert direction_calls == 2
    assert "resume" in commands[1]
    assert "resume" in commands[2]
    assert result["evidence_quality_score"]["support_coverage"]["code"] >= 2
    assert result["evidence_retrieval_trace"]["followup_rounds"][0]["selected_ref_count"] >= 1


def test_gpu_selector_auto_limits_to_six(monkeypatch) -> None:
    snapshot = [
        {"index": idx, "memory_total_mb": 80000, "memory_free_mb": 10000 + idx * 1000, "memory_used_mb": 0, "utilization_gpu": idx}
        for idx in range(8)
    ]
    monkeypatch.setattr(ExperimentRunner, "_gpu_snapshot", staticmethod(lambda: snapshot))
    selection = ExperimentRunner({"experiment": {"gpu_policy": {"max_gpus": 6, "min_free_mb": 0}}, "c2c": {"small_loop": {"gpu_ids": "auto"}}}).select_gpus()
    assert selection.selected_ids == [7, 6, 5, 4, 3, 2]
    assert len(selection.selected_ids) == 6


def test_c2c_preflight_repairs_broken_model_symlink(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    model_root = tmp_path / "models"
    broken = model_root / "Qwen3-0.6B"
    model_root.mkdir()
    broken.symlink_to(model_root / "missing", target_is_directory=True)
    hf_home = tmp_path / "hf_home"
    snapshot = hf_home / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-0.6B" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: hf_home))
    monkeypatch.setattr(C2CAdapter, "_offline_model_load_errors", staticmethod(lambda model_path: []))

    config = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "env_python": "/usr/bin/python3",
            "model_map": {"Qwen/Qwen3-0.6B": str(broken)},
            "datasets": ["mmlu-redux"],
            "small_loop": {"eval_datasets": ["mmlu-redux"], "strict_dataset_cache": False},
        },
        "experiment": {"gpu_policy": {"max_gpus": 1}},
    }
    adapter = C2CAdapter(tmp_path / "project", config)
    candidate = {"id": "idea", "title": "Idea"}
    run_spec = adapter.materialize_candidate_configs(candidate)
    result = adapter.preflight(run_spec)
    assert result["status"] == "ok"
    assert broken.resolve() == snapshot
    assert any(action["status"] == "ok" for action in result["recovery_actions"])


def test_c2c_materializes_candidate_config_overrides(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "env_python": "/usr/bin/python3",
            "datasets": ["mmlu-redux"],
            "small_loop": {"eval_datasets": ["mmlu-redux"], "train_samples": 1, "gpu_ids": [0]},
        }
    }
    adapter = C2CAdapter(tmp_path / "project", config)
    candidate = {
        "id": "override_idea",
        "experiment_contract": {
            "config_overrides": {
                "train": {
                    "model": {
                        "soft_alignment_top_k": 2,
                        "soft_alignment_confidence_floor": 0.2,
                    }
                },
                "eval": {
                    "model": {
                        "rosetta_config": {
                            "soft_alignment_top_k": 2,
                            "soft_alignment_confidence_floor": 0.2,
                        }
                    }
                },
            }
        },
    }
    run_spec = adapter.materialize_candidate_configs(candidate)
    train = json.loads(Path(run_spec["train_config"]).read_text(encoding="utf-8"))
    eval_cfg = yaml.safe_load(next(iter(run_spec["eval_configs"].values())).read_text(encoding="utf-8"))

    assert run_spec["has_executable_change"] is True
    assert train["model"]["soft_alignment_top_k"] == 2
    assert train["model"]["soft_alignment_confidence_floor"] == 0.2
    assert eval_cfg["model"]["rosetta_config"]["soft_alignment_top_k"] == 2
    assert eval_cfg["model"]["rosetta_config"]["soft_alignment_confidence_floor"] == 0.2


def test_c2c_materialized_train_configs_disable_wandb_without_service_token(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "env_python": "/usr/bin/python3",
            "datasets": ["mmlu-redux"],
            "small_loop": {
                "eval_datasets": ["mmlu-redux"],
                "train_samples": 1,
                "gpu_ids": [0],
                "proxy_screen": {
                    "enabled": True,
                    "train_samples": 2,
                    "eval_datasets": ["mmlu-redux"],
                    "per_device_train_batch_size": 1,
                },
            },
        }
    }
    adapter = C2CAdapter(tmp_path / "project", config)
    run_spec = adapter.materialize_candidate_configs({"id": "idea"})
    baseline_spec = adapter.materialize_proxy_baseline_configs()

    train = json.loads(Path(run_spec["train_config"]).read_text(encoding="utf-8"))
    proxy_train = json.loads(Path(run_spec["proxy_screen"]["train_config"]).read_text(encoding="utf-8"))
    baseline_train = json.loads(Path(baseline_spec["train_config"]).read_text(encoding="utf-8"))

    for payload in [train, proxy_train, baseline_train]:
        wandb_config = payload["output"]["wandb_config"]
        assert wandb_config["mode"] == "disabled"
        assert wandb_config["entity"] is None
    for payload in [proxy_train, baseline_train]:
        assert payload["training"]["per_device_train_batch_size"] == 1
        assert payload["training"]["gradient_accumulation_steps"] == 1

    combined_commands = "\n".join(
        [
            run_spec["commands"]["train"],
            run_spec["proxy_screen"]["commands"]["train"],
            baseline_spec["commands"]["train"],
        ]
    )
    assert "WANDB_DISABLED=true" in combined_commands
    assert "WANDB_SERVICE=" not in combined_commands
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in combined_commands


def test_c2c_materializes_proxy_activation_smoke_ablation_config(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "env_python": "/usr/bin/python3",
            "datasets": ["mmlu-redux", "ai2-arc"],
            "small_loop": {
                "eval_datasets": ["mmlu-redux", "ai2-arc"],
                "train_samples": 1,
                "gpu_ids": [0],
                "proxy_screen": {
                    "enabled": True,
                    "eval_limit": 7,
                    "eval_datasets": ["mmlu-redux", "ai2-arc"],
                    "activation_smoke": {"enabled": True, "max_datasets": 1},
                },
            },
        }
    }
    adapter = C2CAdapter(tmp_path / "project", config)
    selection = ExperimentRunner(config).select_gpus({"gpu_ids": [0], "max_gpus": 1})
    candidate = {
        "id": "activation_config",
        "experiment_contract": {
            "ablation_switch": "disable_mechanism",
            "config_overrides": {
                "eval": {"model": {"rosetta_config": {"mechanism_enabled": True}}},
            },
        },
    }

    run_spec = adapter.materialize_candidate_configs(candidate, selection)
    smoke = adapter.materialize_proxy_activation_smoke_configs(candidate, run_spec, selection)
    disabled_eval = yaml.safe_load(Path(smoke["eval_configs"]["mmlu-redux"]).read_text(encoding="utf-8"))

    assert smoke["status"] == "materialized"
    assert smoke["datasets"] == ["mmlu-redux"]
    assert disabled_eval["model"]["rosetta_config"]["checkpoints_dir"] == "local/auto_research_runs/activation_config/proxy/checkpoints/final"
    assert disabled_eval["model"]["rosetta_config"]["disable_mechanism"] is True
    assert disabled_eval["model"]["rosetta_config"]["mechanism_enabled"] is True
    assert disabled_eval["output"]["output_dir"] == "local/auto_research_runs/activation_config/proxy/activation_smoke_disabled/results/mmlu-redux"
    assert disabled_eval["eval"]["limit"] == 7


def test_c2c_proxy_batch_auto_uses_gpu_memory(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    snapshot = [
        {"index": 0, "memory_total_mb": 24564, "memory_free_mb": 24000, "memory_used_mb": 0, "utilization_gpu": 0},
        {"index": 1, "memory_total_mb": 24564, "memory_free_mb": 24000, "memory_used_mb": 0, "utilization_gpu": 0},
    ]
    monkeypatch.setattr(ExperimentRunner, "_gpu_snapshot", staticmethod(lambda: snapshot))
    config = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "env_python": "/usr/bin/python3",
            "datasets": ["mmlu-redux"],
            "small_loop": {
                "eval_datasets": ["mmlu-redux"],
                "gpu_ids": [0, 1],
                "proxy_screen": {"enabled": True, "train_samples": 2, "eval_datasets": ["mmlu-redux"]},
            },
        }
    }
    adapter = C2CAdapter(tmp_path / "project", config)
    selection = ExperimentRunner(config).select_gpus({"gpu_ids": [0, 1], "max_gpus": 2})
    run_spec = adapter.materialize_candidate_configs({"id": "idea"}, selection)
    proxy_train = json.loads(Path(run_spec["proxy_screen"]["train_config"]).read_text(encoding="utf-8"))
    baseline = adapter.materialize_proxy_baseline_configs(selection)
    baseline_train = json.loads(Path(baseline["train_config"]).read_text(encoding="utf-8"))

    assert proxy_train["training"]["per_device_train_batch_size"] == 2
    assert baseline_train["training"]["per_device_train_batch_size"] == 2


def test_runner_select_gpus_filters_busy_cards_when_requested(monkeypatch) -> None:
    snapshot = [
        {"index": 0, "memory_total_mb": 24564, "memory_free_mb": 24000, "memory_used_mb": 0, "utilization_gpu": 98},
        {"index": 1, "memory_total_mb": 24564, "memory_free_mb": 18000, "memory_used_mb": 6500, "utilization_gpu": 5},
        {"index": 2, "memory_total_mb": 24564, "memory_free_mb": 9000, "memory_used_mb": 15500, "utilization_gpu": 0},
    ]
    monkeypatch.setattr(ExperimentRunner, "_gpu_snapshot", staticmethod(lambda: snapshot))

    selection = ExperimentRunner({"experiment": {"gpu_policy": {"max_gpus": 2}}}).select_gpus(
        {"gpu_ids": "auto", "max_gpus": 2, "min_free_mb": 8192, "max_utilization_gpu": 40, "respect_resource_filters": True}
    )

    assert selection.selected_ids == [1, 2]
    assert selection.reason == "auto_selected_by_free_memory"


def test_runner_select_gpus_can_disable_busy_card_fallback(monkeypatch) -> None:
    snapshot = [
        {"index": 0, "memory_total_mb": 24564, "memory_free_mb": 24000, "memory_used_mb": 0, "utilization_gpu": 98},
    ]
    monkeypatch.setattr(ExperimentRunner, "_gpu_snapshot", staticmethod(lambda: snapshot))

    selection = ExperimentRunner({"experiment": {"gpu_policy": {"max_gpus": 1}}}).select_gpus(
        {
            "gpu_ids": "auto",
            "max_gpus": 1,
            "min_free_mb": 8192,
            "max_utilization_gpu": 40,
            "respect_resource_filters": True,
            "disable_resource_fallback": True,
        }
    )

    assert selection.selected_ids == []
    assert selection.reason == "auto_selected_by_free_memory"


def test_runner_select_gpus_can_disable_explicit_busy_card_fallback(monkeypatch) -> None:
    snapshot = [
        {"index": 0, "memory_total_mb": 24564, "memory_free_mb": 24000, "memory_used_mb": 0, "utilization_gpu": 98},
        {"index": 1, "memory_total_mb": 24564, "memory_free_mb": 24000, "memory_used_mb": 0, "utilization_gpu": 0},
    ]
    monkeypatch.setattr(ExperimentRunner, "_gpu_snapshot", staticmethod(lambda: snapshot))

    selection = ExperimentRunner({"experiment": {"gpu_policy": {"max_gpus": 1}}}).select_gpus(
        {
            "gpu_ids": [0],
            "max_gpus": 1,
            "min_free_mb": 8192,
            "max_utilization_gpu": 40,
            "respect_resource_filters": True,
            "disable_resource_fallback": True,
        }
    )

    assert selection.selected_ids == []
    assert selection.reason == "explicit_gpu_ids_no_resource_match"


def test_c2c_proxy_gpu_policy_defaults_to_single_clean_gpu(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    snapshot = [
        {"index": 0, "memory_total_mb": 24564, "memory_free_mb": 24000, "memory_used_mb": 0, "utilization_gpu": 95},
        {"index": 1, "memory_total_mb": 24564, "memory_free_mb": 21000, "memory_used_mb": 3000, "utilization_gpu": 0},
        {"index": 2, "memory_total_mb": 24564, "memory_free_mb": 20000, "memory_used_mb": 4000, "utilization_gpu": 0},
    ]
    monkeypatch.setattr(ExperimentRunner, "_gpu_snapshot", staticmethod(lambda: snapshot))
    config = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "env_python": "/usr/bin/python3",
            "datasets": ["mmlu-redux"],
            "small_loop": {
                "eval_datasets": ["mmlu-redux"],
                "gpu_ids": "auto",
                "proxy_screen": {"enabled": True, "eval_datasets": ["mmlu-redux"], "train_samples": 2},
            },
        },
        "experiment": {"gpu_policy": {"max_gpus": 3, "min_free_mb": 8192, "max_utilization_gpu": 40}},
    }
    project_root = tmp_path / "project"
    context = AgentContext(project_root, config, ArtifactManager(project_root), ModelClient(config, project_root=project_root))
    agent = ExperimentAgent(context)

    full_selection = ExperimentRunner(config).select_gpus({"gpu_ids": "auto", "max_gpus": 3, "min_free_mb": 8192, "max_utilization_gpu": 40})
    proxy_selection = agent._select_c2c_proxy_gpus(agent._c2c_proxy_gpu_policy(execution={}))
    run_spec = C2CAdapter(project_root, config).materialize_candidate_configs({"id": "idea"}, full_selection, proxy_gpu_selection=proxy_selection)

    assert full_selection.selected_ids == [1, 2]
    assert proxy_selection.selected_ids == [1]
    assert "CUDA_VISIBLE_DEVICES=1,2" in run_spec["commands"]["train"]
    assert "CUDA_VISIBLE_DEVICES=1 " in run_spec["proxy_screen"]["commands"]["train"]
    assert "torch.distributed.run" not in run_spec["proxy_screen"]["commands"]["train"]
    assert all("CUDA_VISIBLE_DEVICES=1,2" not in command for command in run_spec["commands"]["preflight"])
    assert all("CUDA_VISIBLE_DEVICES=1 " not in command for command in run_spec["commands"]["preflight"])
    assert any("--no-cov" in command for command in run_spec["commands"]["preflight"])


def test_c2c_full_train_resource_policy_sets_batch_grad_and_lr(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "recipe/train_recipe/C2C_0.6+0.5.json").write_text(
        json.dumps(
            {
                "output": {},
                "data": {"kwargs": {}},
                "training": {
                    "learning_rate": 1e-4,
                    "per_device_train_batch_size": 4,
                    "gradient_accumulation_steps": 8,
                },
                "model": {},
            }
        ),
        encoding="utf-8",
    )
    snapshot = [
        {"index": 5, "memory_total_mb": 24564, "memory_free_mb": 24000, "memory_used_mb": 0, "utilization_gpu": 0},
        {"index": 6, "memory_total_mb": 24564, "memory_free_mb": 17000, "memory_used_mb": 7000, "utilization_gpu": 0},
    ]
    monkeypatch.setattr(ExperimentRunner, "_gpu_snapshot", staticmethod(lambda: snapshot))
    config = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "env_python": "/usr/bin/python3",
            "datasets": ["mmlu-redux"],
            "small_loop": {
                "eval_datasets": ["mmlu-redux"],
                "gpu_ids": "auto",
                "train_resource_policy": {
                    "enabled": True,
                    "per_device_train_batch_size": "auto",
                    "reference_per_device_train_batch_size": 4,
                    "reference_gradient_accumulation_steps": 8,
                    "reference_num_gpus": 1,
                    "learning_rate_scale": "effective_batch_ratio",
                },
                "proxy_screen": {"enabled": False},
            },
        },
        "experiment": {"gpu_policy": {"max_gpus": 2, "min_free_mb": 8192, "max_utilization_gpu": 40}},
    }
    adapter = C2CAdapter(tmp_path / "project", config)
    selection = ExperimentRunner(config).select_gpus({"gpu_ids": "auto", "max_gpus": 2, "min_free_mb": 8192, "max_utilization_gpu": 40})

    run_spec = adapter.materialize_candidate_configs({"id": "idea"}, selection)
    train = json.loads(Path(run_spec["train_config"]).read_text(encoding="utf-8"))

    assert selection.selected_ids == [5, 6]
    assert train["training"]["per_device_train_batch_size"] == 3
    assert train["training"]["gradient_accumulation_steps"] == 6
    assert train["training"]["learning_rate"] == pytest.approx(1.125e-4)
    assert run_spec["train_resource_adjustment"]["selected_gpu_free_mb"] == 17000
    assert run_spec["train_resource_adjustment"]["effective_batch_size"] == 36


def test_c2c_proxy_batch_explicit_override_wins(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    monkeypatch.setattr(
        ExperimentRunner,
        "_gpu_snapshot",
        staticmethod(lambda: [{"index": 0, "memory_total_mb": 24564, "memory_free_mb": 24000, "memory_used_mb": 0, "utilization_gpu": 0}]),
    )
    config = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "env_python": "/usr/bin/python3",
            "datasets": ["mmlu-redux"],
            "small_loop": {
                "eval_datasets": ["mmlu-redux"],
                "gpu_ids": [0],
                "proxy_screen": {
                    "enabled": True,
                    "train_samples": 2,
                    "eval_datasets": ["mmlu-redux"],
                    "per_device_train_batch_size": 1,
                },
            },
        }
    }
    adapter = C2CAdapter(tmp_path / "project", config)
    selection = ExperimentRunner(config).select_gpus({"gpu_ids": [0], "max_gpus": 1})
    run_spec = adapter.materialize_candidate_configs({"id": "idea"}, selection)
    proxy_train = json.loads(Path(run_spec["proxy_screen"]["train_config"]).read_text(encoding="utf-8"))

    assert proxy_train["training"]["per_device_train_batch_size"] == 1


def test_c2c_materialization_localizes_runtime_model_literals(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    dataset_adapter = repo / "rosetta/train/dataset_adapters.py"
    dataset_adapter.parent.mkdir(parents=True, exist_ok=True)
    dataset_adapter.write_text(
        'from transformers import AutoTokenizer\n'
        'TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")\n',
        encoding="utf-8",
    )
    local_model = tmp_path / "models/Qwen3-0.6B"
    local_model.mkdir(parents=True)
    config = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "env_python": "/usr/bin/python3",
            "model_map": {"Qwen/Qwen3-0.6B": str(local_model)},
            "datasets": ["mmlu-redux"],
            "small_loop": {"eval_datasets": ["mmlu-redux"], "train_samples": 1, "gpu_ids": [0]},
        }
    }

    run_spec = C2CAdapter(tmp_path / "project", config).materialize_candidate_configs({"id": "idea"})

    updated = dataset_adapter.read_text(encoding="utf-8")
    assert "Qwen/Qwen3-0.6B" not in updated
    assert str(local_model) in updated
    assert run_spec["runtime_localization"]["status"] == "ok"
    assert run_spec["runtime_localization"]["files"][0]["path"] == "rosetta/train/dataset_adapters.py"


def test_experiment_runner_preserves_command_output_head_and_tail(tmp_path: Path) -> None:
    runner = ExperimentRunner({})
    command = (
        "python - <<'PY'\n"
        "import sys\n"
        "sys.stderr.write('TRACEBACK_START\\n' + 'x' * 15000 + '\\nROOT_CAUSE\\n')\n"
        "raise SystemExit(1)\n"
        "PY"
    )

    result = runner.run_step(name="fail", command=command, working_dir=tmp_path)
    stderr = result["attempts"][0]["stderr"]

    assert result["status"] == "failed"
    assert "TRACEBACK_START" in stderr
    assert "ROOT_CAUSE" in stderr
    assert "truncated" in stderr


def test_experiment_runner_times_out_process_group(tmp_path: Path) -> None:
    runner = ExperimentRunner({})
    command = (
        "python - <<'PY'\n"
        "import time\n"
        "print('started', flush=True)\n"
        "time.sleep(5)\n"
        "PY"
    )

    result = runner.run_step(name="slow", command=command, working_dir=tmp_path, retry_policy={"timeout_seconds": 1})
    attempt = result["attempts"][0]

    assert result["status"] == "failed"
    assert result["returncode"] == 124
    assert attempt["timed_out"] is True
    assert attempt["timeout_seconds"] == 1
    assert "timed out" in attempt["stderr"]


def test_c2c_debate_avoids_failed_feedback_ideas(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=True)
    config["ideation"] = {"debate": {"enabled": True, "rounds": 1}}
    context = AgentContext(tmp_path / "workspace" / "p", config, ArtifactManager(tmp_path / "workspace" / "p"), ModelClient(config, project_root=tmp_path))
    feedback = [
        {
            "kind": "c2c_feedback_summary",
            "failed_idea_ids": [
                "entropy_calibrated_span_gate",
                "headwise_alignment_confidence",
                "length_aware_topk_alignment",
            ],
            "failed_titles": [
                "Entropy-calibrated span confidence gate",
                "Headwise alignment confidence modulation",
                "Length-aware top-k soft alignment",
            ],
            "avoid_repeat_rules": ["Do not repeat this mechanism without addressing mmlu-redux regression."],
            "failure_modes": ["mmlu_regression"],
            "dataset_regressions": {"mmlu-redux": 2.7},
            "summary_text": "latest=c2c_failure_feedback:not_viable | reason=mmlu-redux regression",
        }
    ]

    debate = MultiAgentReasoningService(context).run_c2c_debate(
        topic="cross tokenizer cache",
        repo_card={},
        paper_cards=[],
        rebuttal_matrix={"top_concerns": ["failure_modes_ood"]},
        negative_memory={},
        baseline={"name": "base", "mean": 50.0, "datasets": {}},
        feedback=feedback,
    )
    idea_ids = {idea["id"] for idea in debate["selected_ideas"]}

    assert "entropy_calibrated_span_gate" not in idea_ids
    assert "headwise_alignment_confidence" not in idea_ids
    assert "length_aware_topk_alignment" not in idea_ids
    assert "verifier_guided_cache_acceptance" in idea_ids
    assert debate["negative_constraints"]["forbidden_idea_ids"]
    assert debate["negative_constraints"]["failure_feedback_rules"] == [
        "Do not repeat this mechanism without addressing mmlu-redux regression."
    ]
    assert debate["negative_constraints"]["failure_modes"] == ["mmlu_regression"]
    selected = debate["selected_ideas"][0]
    assert selected["failure_feedback_refs"]
    assert selected["failure_feedback_refs"][0]["source_type"] == "failure_feedback"
    assert "mmlu-redux regression" in selected["failure_feedback_refs"][0]["snippet"]
    assert selected["novelty_gate"]["status"] == "pass"
    assert selected["implementation_scope_gate"]["status"] == "pass"


def test_c2c_debate_emits_decision_chain(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=True)
    config["ideation"] = {"debate": {"enabled": True, "rounds": 1}}
    context = AgentContext(tmp_path / "workspace" / "p", config, ArtifactManager(tmp_path / "workspace" / "p"), ModelClient(config, project_root=tmp_path))

    debate = MultiAgentReasoningService(context).run_c2c_debate(
        topic="cross tokenizer cache",
        repo_card={},
        paper_cards=[],
        rebuttal_matrix={"top_concerns": ["failure_modes_ood"]},
        negative_memory={},
        baseline={"name": "base", "mean": 50.0, "datasets": {}},
        feedback=[],
    )

    assert "decision_chain" in debate
    assert debate["decision_chain"]["evidence"]
    assert debate["decision_chain"]["counterevidence"]
    assert debate["decision_chain"]["conclusion"]
    assert debate["selected_ideas"][0]["decision_chain"]["evidence"]
    assert debate["selected_ideas"][0]["decision_chain"]["counterevidence"]
    assert debate["selected_ideas"][0]["evidence_refs"]
    assert isinstance(debate["selected_ideas"][0]["evidence_refs"][0], dict)


def test_c2c_s1_codex_evidence_agent_blocks_without_fallback(monkeypatch, tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {"enabled": True}
    config["agents"] = {"s1_evidence_agent": {"max_json_repairs": 1, "timeout_seconds": 5}}
    project_root = tmp_path / "workspace" / "p"
    project_root.mkdir(parents=True)
    monkeypatch.setattr(literature_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    original_subprocess_run = literature_module.subprocess.run

    def fake_bad_codex(command, **kwargs):
        if not command or command[0] != "codex":
            return original_subprocess_run(command, **kwargs)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("still not json", encoding="utf-8")
        stdout = '{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614174002"}\n' if "resume" not in command else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(literature_module.subprocess, "run", fake_bad_codex)

    result = literature_module._run_s1_codex_evidence_agent(
        project_root=project_root,
        config=config,
        prompt="return valid json",
        max_repairs=1,
        timeout_seconds=5,
    )

    assert result["status"] == "blocked"
    assert result["repair_count"] == 1
    assert "fallback" not in json.dumps(result)
    assert (project_root / "literature/c2c/s1_codex_events.jsonl").exists()


def test_c2c_s1_codex_evidence_agent_stops_on_backend_auth_failure(monkeypatch, tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {"enabled": True}
    config["agents"] = {"s1_evidence_agent": {"max_json_repairs": 2, "timeout_seconds": 5}}
    project_root = tmp_path / "workspace" / "p_auth"
    project_root.mkdir(parents=True)
    monkeypatch.setattr(literature_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    calls = []

    def fake_auth_failed_codex(command, **kwargs):
        calls.append(command)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("", encoding="utf-8")
        stderr = (
            '{"type":"error","message":"unexpected status 401 Unauthorized: Invalid token, '
            'url: https://api.example.test/v1/responses"}\n'
        )
        return SimpleNamespace(returncode=1, stdout="", stderr=stderr)

    monkeypatch.setattr(literature_module.subprocess, "run", fake_auth_failed_codex)

    result = literature_module._run_s1_codex_evidence_agent(
        project_root=project_root,
        config=config,
        prompt="return valid json",
        max_repairs=2,
        timeout_seconds=5,
    )

    assert result["status"] == "blocked"
    assert result["failure_category"] == "llm_authentication"
    assert result["repair_count"] == 0
    assert len(calls) == 1
    assert "invalid token" in result["reason"].lower()
    event_log = (project_root / "literature/c2c/s1_codex_events.jsonl").read_text(encoding="utf-8")
    assert "llm_authentication" in event_log


def test_c2c_s1_codex_evidence_agent_repairs_unresolved_refs(monkeypatch, tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {"enabled": True}
    config["agents"] = {"s1_evidence_agent": {"max_json_repairs": 1, "timeout_seconds": 5}}
    project_root = tmp_path / "workspace" / "p_refs"
    project_root.mkdir(parents=True)
    _write_minimal_s1_ref_catalog(project_root)
    monkeypatch.setattr(literature_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    prompts = []

    def fake_codex(command, **kwargs):
        prompts.append(kwargs.get("input") or "")
        output_path = Path(command[command.index("--output-last-message") + 1])
        payload = _s1_codex_direction_payload()
        if "resume" not in command:
            payload["evidence_bundle"]["items"][0]["chunk_id"] = "missing:chunk"
            payload["selected_ideas"][0]["code_refs"] = [{"source_type": "code", "source_label": "missing/file.py", "claim": "bad ref"}]
            stdout = '{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614174333"}\n'
        else:
            stdout = ""
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(literature_module.subprocess, "run", fake_codex)

    result = literature_module._run_s1_codex_evidence_agent(
        project_root=project_root,
        config=config,
        prompt="return valid json",
        max_repairs=1,
        timeout_seconds=5,
    )

    assert result["status"] == "ok"
    assert result["repair_count"] == 1
    assert result["evidence_ref_report"]["status"] == "pass"
    assert "chunk_id_not_in_chunk_index" in prompts[1]
    assert "code_ref_missing_file_or_symbol" in prompts[1]


def test_c2c_s1_codex_evidence_agent_accepts_artifact_anchor_refs(monkeypatch, tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {"enabled": True}
    config["agents"] = {"s1_evidence_agent": {"max_json_repairs": 1, "timeout_seconds": 5}}
    project_root = tmp_path / "workspace" / "p_anchor_refs"
    project_root.mkdir(parents=True)
    _write_minimal_s1_ref_catalog(project_root)
    for rel in [
        "experiment/results/failure_feedback.json",
        "plan/direction_scorecard.json",
        "intake/c2c/negative_result_memory.json",
    ]:
        path = project_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(literature_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    calls = []

    def fake_codex(command, **kwargs):
        calls.append(command)
        output_path = Path(command[command.index("--output-last-message") + 1])
        payload = _s1_codex_direction_payload()
        payload["evidence_bundle"]["items"].extend(
            [
                {
                    "source_type": "artifact",
                    "source_path": "experiment/results/failure_feedback.json#entry",
                    "summary": "latest failure feedback",
                    "supports": ["direction"],
                    "risks": [],
                },
                {
                    "source_type": "artifact",
                    "source_path": "plan/direction_scorecard.json#candidate_constrained",
                    "summary": "prior direction exhausted",
                    "supports": [],
                    "risks": ["repeat risk"],
                },
            ]
        )
        payload["selected_ideas"][0]["evidence_refs"].append(
            {
                "source_type": "artifact",
                "source_label": "experiment/results/failure_feedback.json#entry",
                "claim": "anchor refs should resolve by artifact path",
            }
        )
        payload["selected_ideas"][0]["counterevidence_refs"].append(
            {
                "source_type": "artifact",
                "source_label": "intake/c2c/negative_result_memory.json#blocked_idea_patterns",
                "claim": "anchor refs should not force JSON repair",
            }
        )
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614174555"}\n',
            stderr="",
        )

    monkeypatch.setattr(literature_module.subprocess, "run", fake_codex)

    result = literature_module._run_s1_codex_evidence_agent(
        project_root=project_root,
        config=config,
        prompt="return valid json",
        max_repairs=1,
        timeout_seconds=5,
    )

    assert result["status"] == "ok"
    assert result["repair_count"] == 0
    assert len(calls) == 1
    assert result["evidence_ref_report"]["status"] == "pass"


def test_c2c_s1_novelty_auditor_rejects_and_revises_direction(monkeypatch, tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {"enabled": True}
    config["agents"] = {
        "s1_evidence_agent": {"max_json_repairs": 0, "timeout_seconds": 5},
        "s1_novelty_auditor": {"enabled": True, "threshold": 0.6, "max_revision_rounds": 1, "timeout_seconds": 5},
    }
    project_root = tmp_path / "workspace" / "p_novelty"
    project_root.mkdir(parents=True)
    _write_minimal_s1_ref_catalog(project_root)
    monkeypatch.setattr(literature_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    prompts: list[str] = []
    s1_call_count = 0
    audit_call_count = 0

    def revised_payload() -> dict:
        payload = _s1_codex_direction_payload()
        payload["direction_decision"]["direction_id"] = "pathology_conditioned_controller"
        payload["direction_decision"]["mechanism_direction"] = "Pathology Conditioned Controller"
        payload["direction_decision"]["mechanism_type"] = "pathology_conditioned_controller"
        payload["direction_decision"]["core_hypothesis"] = "Condition transfer behavior on detected alignment pathology rather than utility routing."
        payload["selected_ideas"][0]["id"] = "pathology_conditioned_controller"
        payload["selected_ideas"][0]["title"] = "Pathology Conditioned Controller"
        payload["selected_ideas"][0]["mechanism_type"] = "pathology_conditioned_controller"
        payload["selected_ideas"][0]["hypothesis"] = "Detect alignment pathology and modulate transfer behavior only when pathology is present."
        return payload

    def fake_codex(command, **kwargs):
        nonlocal s1_call_count, audit_call_count
        prompt = kwargs.get("input") or ""
        prompts.append(prompt)
        output_path = Path(command[command.index("--output-last-message") + 1])
        if "auditing S1 novelty" in prompt:
            audit_call_count += 1
            if audit_call_count == 1:
                output_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "c2c_s1_novelty_audit_v1",
                            "status": "ok",
                            "novelty_score": 0.22,
                            "max_similarity_score": 0.88,
                            "passed": False,
                            "threshold": 0.6,
                            "most_similar_sources": [{"source_type": "shared_memory", "source_id": "mem_old_utility", "similarity_score": 0.88, "overlap": ["mechanism"], "why_similar": "same utility routing mechanism"}],
                            "distinctive_elements": [],
                            "repeated_patterns": ["utility routing"],
                            "revision_guidance": ["Switch away from utility routing toward pathology-conditioned control."],
                            "decision": "revise",
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                output_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "c2c_s1_novelty_audit_v1",
                            "status": "ok",
                            "novelty_score": 0.78,
                            "max_similarity_score": 0.31,
                            "passed": True,
                            "threshold": 0.6,
                            "most_similar_sources": [],
                            "distinctive_elements": ["pathology-conditioned control"],
                            "repeated_patterns": [],
                            "revision_guidance": [],
                            "decision": "pass",
                        }
                    ),
                    encoding="utf-8",
                )
            return SimpleNamespace(returncode=0, stdout='{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614179999"}\n', stderr="")
        s1_call_count += 1
        output_path.write_text(json.dumps(_s1_codex_direction_payload() if s1_call_count == 1 else revised_payload()), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout='{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614170000"}\n', stderr="")

    monkeypatch.setattr(literature_module.subprocess, "run", fake_codex)

    result = literature_module._run_s1_codex_evidence_agent(
        project_root=project_root,
        config=config,
        prompt="return valid json",
        max_repairs=0,
        timeout_seconds=5,
    )

    assert result["status"] == "ok"
    assert s1_call_count == 2
    assert audit_call_count == 2
    assert len(result["novelty_audits"]) == 2
    assert result["novelty_audits"][0]["passed"] is False
    assert result["novelty_audits"][1]["passed"] is True
    assert result["payload"]["direction_decision"]["direction_id"] == "pathology_conditioned_controller"
    assert any("previous S1 direction was rejected" in prompt for prompt in prompts)


def test_c2c_s1_codex_resets_duplicate_direction_session(monkeypatch, tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {"enabled": True}
    config["agents"] = {"s1_evidence_agent": {"duplicate_direction_reset_threshold": 1, "timeout_seconds": 5}}
    project_root = tmp_path / "workspace" / "p_duplicate_s1"
    project_root.mkdir(parents=True)
    _write_minimal_s1_ref_catalog(project_root)
    monkeypatch.setattr(literature_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)

    def fake_codex(command, **kwargs):
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(json.dumps(_s1_codex_direction_payload()), encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614174444"}\n',
            stderr="",
        )

    monkeypatch.setattr(literature_module.subprocess, "run", fake_codex)

    first = literature_module._run_s1_codex_evidence_agent(
        project_root=project_root,
        config=config,
        prompt="return valid json",
        max_repairs=0,
        timeout_seconds=5,
    )
    second = literature_module._run_s1_codex_evidence_agent(
        project_root=project_root,
        config=config,
        prompt="return valid json",
        max_repairs=0,
        timeout_seconds=5,
    )

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert second["session_reset"] is True
    assert second["session_reset_reason"] == "duplicate_direction_streak"
    sessions = yaml.safe_load((project_root / "meta" / "codex_sessions.yaml").read_text(encoding="utf-8"))
    assert "s1:c2c_evidence_direction" not in sessions.get("sessions", {})
    assert sessions["session_reset_history"][-1]["reason"] == "duplicate_direction_streak"


def test_c2c_s2_directional_planner_falls_back_without_real_llm(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {"eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"], "gpu_ids": [0], "max_candidates": 2},
        "allowed_files": ["rosetta/model/aligner.py", "rosetta/model/projector.py", "rosetta/model/wrapper.py", "script/train/SFT_train.py", "test/test_aligner_span_overlap.py"],
    }
    config["agents"] = {"s2_directional_planner": {"resume_enabled": False}}
    config["code_patch"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_s2_planner_fallback", simulate=False)
    ideas = default_c2c_ideas("topic", config["c2c"]["baseline"])
    ArtifactManager(paths.root).write_json("S1_literature", "ideas.json", ideas, artifact_type="ideas", summary="ideas")
    ArtifactManager(paths.root).write_json("S1_literature", "c2c/baseline_evidence.json", config["c2c"]["baseline"], artifact_type="baseline", summary="baseline")

    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    result = PlanAgent(context).run()

    planned = result["plan"]["candidate_ideas"]
    assert planned[0]["id"] == "utility_predicted_cache_routing"
    assert result["plan"]["directional_planning"]["status"] == "fallback_no_real_llm"
    assert (paths.root / "plan" / "s2_planner" / "candidate_pool.json").exists()
    assert (paths.root / "plan" / "s2_planner" / "feedback_context.json").exists()
    assert (paths.root / "plan" / "s2_planner" / "adaptive_policy.json").exists()
    assert (paths.root / "plan" / "s2_planner" / "variant_scorecard.json").exists()
    assert (paths.root / "plan" / "s2_planner" / "score_adjustment_report.json").exists()
    assert (paths.root / "plan" / "s2_planner" / "next_variant.json").exists()
    adaptive_policy = json.loads((paths.root / "plan" / "s2_planner" / "adaptive_policy.json").read_text(encoding="utf-8"))
    scorecard = json.loads((paths.root / "plan" / "s2_planner" / "variant_scorecard.json").read_text(encoding="utf-8"))
    adjustment = json.loads((paths.root / "plan" / "s2_planner" / "score_adjustment_report.json").read_text(encoding="utf-8"))
    assert scorecard["policy_hash"] == adaptive_policy["policy_hash"]
    assert adjustment["selected_variant_id"] == scorecard["selected_variant_id"]
    planner_gate = json.loads((paths.root / "plan" / "s2_planner" / "planner_gate_report.json").read_text(encoding="utf-8"))
    assert planner_gate["gate"] == "pass"


def test_c2c_s2_planner_current_direction_fallback_gets_execution_contract(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {"eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"], "gpu_ids": [0], "max_candidates": 2},
        "allowed_files": ["rosetta/model/aligner.py", "rosetta/model/projector.py", "rosetta/model/wrapper.py"],
    }
    config["agents"] = {"s2_directional_planner": {"resume_enabled": False}}
    config["code_patch"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_s2_direction_contract_fallback", simulate=False)
    s1_direction = {
        "id": "pathology_conditioned_transfer_controller",
        "title": "Pathology-Conditioned Transfer Controller",
        "selected": True,
        "hypothesis": "Condition transfer on alignment pathology buckets.",
        "description": "Use pathology statistics to abstain or attenuate harmful transfer.",
        "mechanism_type": "pathology_conditioned_controller",
        "expected_files": ["rosetta/model/aligner.py", "rosetta/model/projector.py", "rosetta/model/wrapper.py"],
    }
    ArtifactManager(paths.root).write_json("S1_literature", "ideas.json", [s1_direction], artifact_type="ideas", summary="ideas")
    ArtifactManager(paths.root).write_json("S1_literature", "c2c/baseline_evidence.json", config["c2c"]["baseline"], artifact_type="baseline", summary="baseline")

    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    result = PlanAgent(context).run()

    planned = result["plan"]["candidate_ideas"]
    assert planned[0]["id"] == "pathology_conditioned_transfer_controller"
    assert planned[0]["mechanism_type"] == "pathology_conditioned_controller"
    assert planned[0]["novelty_gate"]["status"] == "pass"
    assert planned[0]["implementation_scope_gate"]["status"] == "pass"
    assert planned[0]["experiment_contract"]["ablation_switch"]
    assert planned[0]["experiment_contract"]["config_overrides"]["train"]["model"]["cache_controller_mode"] == "pathology_conditioned_transfer_controller"


def test_c2c_s2_directional_planner_uses_direction_variants(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {"eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"], "gpu_ids": [0], "max_candidates": 2},
        "allowed_files": ["rosetta/model/projector.py", "rosetta/model/wrapper.py", "script/train/SFT_train.py", "test/test_aligner_span_overlap.py"],
    }
    config["agents"] = {"s2_directional_planner": {"resume_enabled": False}}
    config["code_patch"] = {"enabled": False}
    config.setdefault("orchestration", {})["shared_method_memory"] = {
        "enabled": True,
        "path": str(tmp_path / "s2_shared_memory.jsonl"),
        "summary_path": str(tmp_path / "s2_shared_memory.md"),
    }
    shared_memory_entry = {
        "schema_version": "shared_method_failure_memory_v1",
        "memory_id": "mem_s2_proxy_collapse",
        "timestamp": "2026-06-10T00:00:00Z",
        "project_id": "old_project",
        "topic": "cross tokenizer cache",
        "kind": "shared_c2c_method_failure",
        "route": "proxy_rejected_same_direction",
        "failure_class": "method_failure",
        "summary": {
            "failed_idea_ids": ["old_hard_gate"],
            "dragging_datasets": [{"dataset": "mmlu-redux", "regression": 2.0}],
            "avoid_repeat_rules": ["Avoid hard gates that collapse coverage."],
        },
        "entries": [{"id": "old_hard_gate", "decision": "proxy_rejected", "proxy_screen": {"proxy_dataset_deltas": {"mmlu-redux": -2.0}}}],
    }
    Path(config["orchestration"]["shared_method_memory"]["path"]).write_text(json.dumps(shared_memory_entry, ensure_ascii=False) + "\n", encoding="utf-8")
    paths = init_workspace(config, "topic", project_id="proj_s2_planner_variants", simulate=False)
    ideas = default_c2c_ideas("topic", config["c2c"]["baseline"])
    ArtifactManager(paths.root).write_json("S1_literature", "ideas.json", ideas, artifact_type="ideas", summary="ideas")
    ArtifactManager(paths.root).write_json("S1_literature", "c2c/baseline_evidence.json", config["c2c"]["baseline"], artifact_type="baseline", summary="baseline")

    class PlannerLLM(ModelClient):
        def __init__(self, config, project_root=None):
            super().__init__(config, project_root=project_root)
            self.use_real_api = True
            self.prompts = []

        def generate_json_with_schema(self, **kwargs):
            self.prompts.append(kwargs.get("prompt", ""))
            return {
                "planner_summary": "Create two utility-routing implementation variants within the S1 direction.",
                "planning_mode": "same_direction_variant",
                "used_shared_memory_refs": ["mem_s2_proxy_collapse"],
                "variant_candidates": [
                    {
                        "id": "utility_router_soft_residual_variant",
                        "title": "Utility router soft residual variant",
                        "mechanism_axis": "normalization",
                        "integration_point": "wrapper",
                        "control_signal": "utility",
                        "expected_dataset_tradeoff": {"mmlu-redux": "flat", "ai2-arc": "up", "openbookqa": "flat"},
                        "risk_budget": {"max_changed_files": 2, "forbidden_files": ["script/evaluation/*"]},
                        "anti_repeat": "Moves away from projector hard routing toward wrapper residual scaling.",
                        "description": "Keep baseline transfer as the default path and use predicted utility to scale a residual correction rather than adding a hard accept/reject gate.",
                        "motivation": "Previous proxy failures collapsed all datasets, so the variant should preserve baseline coverage while only attenuating harmful residual transfer.",
                        "hypothesis": "A soft residual utility router improves proxy mean without lowering transfer coverage across mmlu-redux, ai2-arc, and openbookqa.",
                        "mechanism_type": "utility_predicted_cache_routing",
                        "paper_claim": "Receiver utility should modulate residual cache transfer instead of replacing the original C2C path.",
                        "why_baseline_fails": "The baseline lacks a downstream utility signal for residual cache injection.",
                        "expected_signature": {"primary": "utility-positive spans keep baseline coverage while harmful residuals shrink", "stats": ["utility_residual_scale", "baseline_transfer_coverage"]},
                        "experiment_contract": {
                            "config_overrides": {
                                "train": {"model": {"cache_routing_mode": "utility_soft_residual", "cache_routing_loss_weight": 0.05}},
                                "eval": {"model": {"rosetta_config": {"cache_routing_mode": "utility_soft_residual"}}},
                            }
                        },
                        "failure_avoidance": ["Do not add another hard gate", "Preserve baseline transfer coverage"],
                        "failure_feedback_refs": [{"source_type": "failure_feedback", "source_label": "proxy collapsed all datasets"}],
                        "used_shared_memory_refs": ["mem_s2_proxy_collapse"],
                    }
                ],
            }

    llm = PlannerLLM(config, project_root=paths.root)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), llm)
    result = PlanAgent(context).run()

    planned = result["plan"]["candidate_ideas"]
    assert result["plan"]["directional_planning"]["status"] == "ok"
    assert result["plan"]["directional_planning"]["memory_entry_count"] == 1
    assert planned[0]["id"] == "utility_router_soft_residual_variant"
    assert planned[0]["mechanism_type"] == "utility_predicted_cache_routing"
    assert planned[0]["selected"] is True
    assert planned[0]["variant_fingerprint"]
    assert planned[0]["s2_variant"]["integration_point"] == "wrapper"
    assert planned[0]["used_shared_memory_refs"] == ["mem_s2_proxy_collapse"]
    assert planned[0]["s2_variant"]["used_shared_memory_refs"] == ["mem_s2_proxy_collapse"]
    assert planned[0]["s2_variant"]["variant_score"]["score"] > 0
    assert planned[0]["s2_planner"]["source"] == "directional_planner"
    assert planned[0]["s2_planner"]["used_shared_memory_refs"] == ["mem_s2_proxy_collapse"]
    assert result["plan"]["directional_planning"]["used_shared_memory_refs"] == ["mem_s2_proxy_collapse"]
    assert result["plan"]["next_variant"]["variant_fingerprint"] == planned[0]["variant_fingerprint"]
    assert result["plan"]["next_variant"]["used_shared_memory_refs"] == ["mem_s2_proxy_collapse"]
    assert (paths.root / "plan" / "next_variant.json").exists()

    saved = json.loads((paths.root / "plan" / "candidate_ideas.json").read_text(encoding="utf-8"))
    assert saved[0]["id"] == "utility_router_soft_residual_variant"
    assert saved[0]["s2_variant"]["variant_fingerprint"] == planned[0]["variant_fingerprint"]
    assert saved[0]["used_shared_memory_refs"] == ["mem_s2_proxy_collapse"]
    variant_artifact = json.loads((paths.root / "plan" / "next_variant.json").read_text(encoding="utf-8"))
    assert variant_artifact["used_shared_memory_refs"] == ["mem_s2_proxy_collapse"]
    assert variant_artifact["next_variant"]["used_shared_memory_refs"] == ["mem_s2_proxy_collapse"]
    memory = json.loads((paths.root / "plan" / "s2_planner_memory.json").read_text(encoding="utf-8"))
    assert memory["entry_count"] == 1
    assert memory["entries"][0]["selected_candidate"]["id"] == "utility_router_soft_residual_variant"
    assert memory["entries"][0]["selected_candidate"]["variant_fingerprint"] == planned[0]["variant_fingerprint"]
    assert memory["entries"][0]["selected_candidate"]["used_shared_memory_refs"] == ["mem_s2_proxy_collapse"]

    second = PlanAgent(context).run()

    assert second["plan"]["directional_planning"]["memory_entry_count"] == 2
    assert "utility_router_soft_residual_variant" in llm.prompts[-1]
    memory = json.loads((paths.root / "plan" / "s2_planner_memory.json").read_text(encoding="utf-8"))
    assert memory["entry_count"] == 2


def test_c2c_s2_adaptive_selector_avoids_failed_integration_after_route_to_s2(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {"eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"], "gpu_ids": [0], "max_candidates": 2},
        "allowed_files": ["rosetta/model/projector.py", "rosetta/model/wrapper.py"],
        "allowed_prefixes": ["rosetta/model"],
        "s2_adaptive_policy": {"enabled": True},
    }
    config["code_patch"] = {"enabled": False}
    config["agents"] = {"s2_directional_planner": {"resume_enabled": False}}
    config["orchestration"]["route_policy"] = {"budgets": {"same_direction_proxy_failures": 2, "same_direction_full_s3_failures": 1}}
    paths = init_workspace(config, "topic", project_id="proj_s2_adaptive_selector", simulate=False)
    direction = {
        "direction_id": "direction_x",
        "title": "Utility routing",
        "mechanism_axis": "routing",
        "integration_point": "projector",
        "control_signal": "utility",
        "hypothesis": "Route cache by utility.",
        "expected_metric_signature": {"primary_metric": "three_dataset_mean"},
        "expected_files": ["rosetta/model/projector.py"],
    }
    ArtifactManager(paths.root).write_json("S1_literature", "direction.json", direction, artifact_type="direction", summary="direction")
    ArtifactManager(paths.root).write_json(
        "S1_literature",
        "ideas.json",
        [
            {
                "id": "direction_x",
                "title": "Utility routing",
                "selected": True,
                "s1_direction_id": "direction_x",
                "mechanism_axis": "routing",
                "integration_point": "projector",
                "control_signal": "utility",
                "expected_files": ["rosetta/model/projector.py"],
            }
        ],
        artifact_type="ideas",
        summary="ideas",
    )
    ArtifactManager(paths.root).write_json("S1_literature", "c2c/baseline_evidence.json", config["c2c"]["baseline"], artifact_type="baseline", summary="baseline")
    write_json(
        paths.root / "meta" / "route_decision.json",
        {
            "decision": "route_to_s2",
            "failure_class": "proxy_false_positive",
            "reason_codes": ["proxy_decision_report_route_hint_return_s2"],
            "budget_effects": {"consumes_same_direction_attempt": True},
        },
    )
    write_json(
        paths.root / "meta" / "attempt_ledger.json",
        {
            "schema_version": "c2c_attempt_ledger_v1",
            "project_id": paths.root.name,
            "records": [],
            "counters": {"by_direction": {"direction_x": {"proxy_failures": 1, "full_s3_failures": 0, "patch_repairs": 0, "resource_retries": 0}}},
        },
    )

    class PlannerLLM(ModelClient):
        def __init__(self, config, project_root=None):
            super().__init__(config, project_root=project_root)
            self.use_real_api = True

        def generate_json_with_schema(self, **kwargs):
            del kwargs
            return {
                "planner_summary": "Compare old and new integration points.",
                "planning_mode": "same_direction_variant",
                "used_shared_memory_refs": [],
                "variant_candidates": [
                    {
                        "id": "old_projector_variant",
                        "title": "Old projector variant",
                        "mechanism_axis": "routing",
                        "integration_point": "projector",
                        "control_signal": "utility",
                        "hypothesis": "Keep projector utility routing.",
                        "expected_files": ["rosetta/model/projector.py"],
                        "ablation_switch": "disable_old_projector",
                        "experiment_contract": {"expected_files": ["rosetta/model/projector.py"], "ablation_switch": "disable_old_projector"},
                    },
                    {
                        "id": "new_wrapper_variant",
                        "title": "New wrapper variant",
                        "mechanism_axis": "routing",
                        "integration_point": "wrapper",
                        "control_signal": "utility",
                        "hypothesis": "Move utility routing to wrapper residual scaling.",
                        "expected_files": ["rosetta/model/wrapper.py"],
                        "ablation_switch": "disable_new_wrapper",
                        "experiment_contract": {"expected_files": ["rosetta/model/wrapper.py"], "ablation_switch": "disable_new_wrapper"},
                    },
                ],
            }

    context = AgentContext(paths.root, config, ArtifactManager(paths.root), PlannerLLM(config, project_root=paths.root))
    result = PlanAgent(context).run()

    assert result["plan"]["next_variant"]["id"] == "new_wrapper_variant"
    scorecard = json.loads((paths.root / "plan" / "s2_planner" / "variant_scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["selected_variant_id"] == "new_wrapper_variant"
    old_row = next(row for row in scorecard["ranking"] if row["variant_id"] == "old_projector_variant")
    assert old_row["components"]["route_history_prior"] < 0


def test_c2c_implementation_failure_reruns_only_s2_5_patch_repair(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {"eval_datasets": ["mmlu-redux"], "gpu_ids": [0], "max_candidates": 1},
        "allowed_files": ["rosetta/model/projector.py", "rosetta/model/wrapper.py"],
    }
    config["agents"] = {"s2_directional_planner": {"resume_enabled": True}}
    config["code_patch"] = {
        "enabled": True,
        "backend": "mock_codex",
        "max_candidates": 1,
        "validation": {
            "require_py_compile": True,
            "require_targeted_tests": False,
            "runtime_smoke": {"enabled": False},
            "mechanism_self_review": {"enabled": False},
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_s2_5_patch_only", simulate=False)
    ideas = default_c2c_ideas("topic", config["c2c"]["baseline"])
    ArtifactManager(paths.root).write_json("S1_literature", "ideas.json", ideas, artifact_type="ideas", summary="ideas")
    ArtifactManager(paths.root).write_json("S1_literature", "c2c/baseline_evidence.json", config["c2c"]["baseline"], artifact_type="baseline", summary="baseline")
    existing_candidate = dict(ideas[0])
    existing_candidate["selected"] = True
    existing_candidate["variant_fingerprint"] = "vf-implementation-repair"
    other_candidate = dict(ideas[1])
    other_candidate["id"] = "must_not_patch"
    other_candidate["selected"] = False
    write_json(paths.root / "plan" / "candidate_ideas.json", [existing_candidate, other_candidate])
    write_json(
        paths.root / "plan" / "s2_5_repair_dispatch.json",
        {
            "schema_version": "c2c_s2_5_only_repair_dispatch_v1",
            "mode": "s2_5_only_implementation_repair",
            "repair_lane": "s2_5_only_implementation_repair",
            "selected_candidate_id": existing_candidate["id"],
            "variant_fingerprint": "vf-implementation-repair",
            "same_candidate_required": True,
            "same_variant_fingerprint_required": True,
            "reuse_persistent_codex_session": True,
            "do_not_replan_method": True,
            "repair_until": "patch_eligible_for_s3_or_implementation_blocked",
            "changed_files": ["rosetta/model/projector.py"],
            "implementation_failure_signals": ["mechanism_activation_wiring_failed", "forward_missing_switch_read"],
            "activation_forward_probe_diagnostics": {"switch_seen_by_forward": False, "projector_output_identical": True},
            "tensor_checks": {"identical_tensors": ["projector_output"], "changed_tensors": []},
            "patch_manifest": {"selected_candidate_id": existing_candidate["id"], "status": "no_valid_patch"},
            "performance_feedback_path": "plan/performance_feedback.json",
            "patch_manifest_path": "plan/code_patches/patch_manifest.json",
        },
    )
    write_json(
        paths.root / "plan" / "performance_feedback.json",
        {
            "summary": {
                "failure_class": "implementation_failure",
                "recommended_s2_action": "patch_repair",
                "does_not_consume_same_direction_attempt": True,
            },
            "reason": "runtime_smoke:mechanism_activation_wiring failed",
        },
    )
    write_json(
        paths.root / "plan" / "s2_planner_memory.json",
        {"schema_version": "c2c_s2_planner_memory_v1", "entries": [{"selected_candidate": {"id": "old"}}]},
    )

    class PlannerMustNotRun(ModelClient):
        def __init__(self, config, project_root=None):
            super().__init__(config, project_root=project_root)
            self.use_real_api = True

        def generate_json_with_schema(self, **kwargs):
            raise AssertionError("S2 planner must be skipped for implementation_failure patch-only repair")

    class PatchBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            assert implementation_contract["candidate_id"] == existing_candidate["id"]
            assert implementation_contract["variant_fingerprint"] == "vf-implementation-repair"
            assert implementation_contract["previous_failure"]["proxy_screen"]["reason"] == "runtime_smoke:mechanism_activation_wiring failed"
            previous_failure = implementation_contract["previous_failure"]
            dispatch = previous_failure["s2_5_repair_dispatch"]
            assert dispatch["selected_candidate_id"] == existing_candidate["id"]
            assert dispatch["tensor_checks"]["identical_tensors"] == ["projector_output"]
            contract = previous_failure["proxy_effect_repair_contract"]
            assert contract["mode"] == "s2_5_only_implementation_repair"
            assert contract["force_new_codex_session"] is False
            assert contract["reuse_persistent_codex_session"] is True
            assert contract["same_candidate_required"] is True
            assert contract["same_variant_fingerprint_required"] is True
            assert "forward_missing_switch_read" in contract["implementation_failure_signals"]
            assert edit_policy.allowed("rosetta/model/projector.py", repo_root=temp_repo)
            (temp_repo / "rosetta/model/projector.py").write_text("VALUE = 's2.5 repaired projector'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "S2.5 patch-only implementation repair."}

    monkeypatch.setattr("auto_research.agents.plan.CodePatchAgent", lambda project_root, config, artifacts: CodePatchAgent(project_root, config, artifacts, backend=PatchBackend()))
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), PlannerMustNotRun(config, project_root=paths.root))
    result = PlanAgent(context).run()

    assert result["plan"]["s2_5_patch_only_repair"]["enabled"] is True
    assert result["plan"]["s2_5_patch_only_repair"]["repair_lane"] == "s2_5_only_implementation_repair"
    assert result["plan"]["s2_5_patch_only_repair"]["skips_s2_planner"] is True
    assert result["plan"]["s2_5_patch_only_repair"]["reuse_persistent_codex_session"] is True
    assert (paths.root / "plan" / "s2_5_patch_only_repair.json").exists()
    assert (paths.root / "plan" / "code_patches" / existing_candidate["id"] / "patch.json").exists()
    assert not (paths.root / "plan" / "code_patches" / other_candidate["id"] / "patch.json").exists()
    plan_yaml = yaml.safe_load((paths.root / "plan" / "plan.yaml").read_text(encoding="utf-8"))
    assert plan_yaml["selected_idea"]["id"] == existing_candidate["id"]
    assert plan_yaml["s2_5_patch_only_repair"]["patch_eligible_for_s3"] is True
    assert len(plan_yaml["candidate_ideas"]) == 1
    assert plan_yaml["candidate_ideas"][0]["id"] == existing_candidate["id"]
    manifest = json.loads((paths.root / "plan" / "code_patches" / "patch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["selected_candidate_id"] == existing_candidate["id"]
    assert (paths.root / "plan" / "code_patches" / "implementation_contract.json").exists()
    patch_gate = json.loads((paths.root / "plan" / "code_patches" / "patch_gate_report.json").read_text(encoding="utf-8"))
    assert patch_gate["gate"] == "pass"
    assert patch_gate["variant_id"] == existing_candidate["id"]
    patch_only_record = json.loads((paths.root / "plan" / "s2_5_patch_only_repair.json").read_text(encoding="utf-8"))
    assert patch_only_record["repair_lane"] == "s2_5_only_implementation_repair"
    assert patch_only_record["patch_eligible_for_s3"] is True
    assert patch_only_record["implementation_blocked"] is False
    assert patch_only_record["selected_candidate_id"] == existing_candidate["id"]
    assert patch_only_record["requested_variant_fingerprint"] == "vf-implementation-repair"
    memory = json.loads((paths.root / "plan" / "s2_planner_memory.json").read_text(encoding="utf-8"))
    assert memory["entries"] == [{"selected_candidate": {"id": "old"}}]


def test_c2c_implementation_failure_missing_candidate_blocks_without_s2_planner(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {"eval_datasets": ["mmlu-redux"], "gpu_ids": [0], "max_candidates": 1},
        "allowed_files": ["rosetta/model/projector.py"],
    }
    config["agents"] = {"s2_directional_planner": {"resume_enabled": True}}
    config["code_patch"] = {"enabled": True, "backend": "mock_codex"}
    paths = init_workspace(config, "topic", project_id="proj_s2_5_missing_candidate", simulate=False)
    ideas = default_c2c_ideas("topic", config["c2c"]["baseline"])
    existing_candidate = dict(ideas[0])
    existing_candidate["id"] = "actual_candidate"
    existing_candidate["selected"] = True
    write_json(paths.root / "plan" / "candidate_ideas.json", [existing_candidate])
    write_json(
        paths.root / "plan" / "s2_5_repair_dispatch.json",
        {
            "mode": "s2_5_only_implementation_repair",
            "selected_candidate_id": "missing_candidate",
            "variant_fingerprint": "missing-fingerprint",
            "reuse_persistent_codex_session": True,
        },
    )
    write_json(
        paths.root / "plan" / "performance_feedback.json",
        {
            "summary": {
                "failure_class": "implementation_failure",
                "recommended_s2_action": "patch_repair",
                "does_not_consume_same_direction_attempt": True,
            },
            "reason": "target candidate missing after S3 implementation failure",
        },
    )

    class PlannerMustNotRun(ModelClient):
        def __init__(self, config, project_root=None):
            super().__init__(config, project_root=project_root)
            self.use_real_api = True

        def generate_json_with_schema(self, **kwargs):
            raise AssertionError("S2 planner must not run for missing-candidate implementation repair")

    class PatchMustNotRun:
        def run(self, plan, patch_ideas):
            raise AssertionError("S2.5 patch generation must not run without the locked candidate")

    monkeypatch.setattr("auto_research.agents.plan.CodePatchAgent", lambda project_root, config, artifacts: PatchMustNotRun())
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), PlannerMustNotRun(config, project_root=paths.root))
    result = PlanAgent(context).run()

    assert result["plan"]["s2_5_patch_only_repair"]["status"] == "implementation_blocked"
    assert result["plan"]["s2_5_patch_only_repair"]["patch_eligible_for_s3"] is False
    assert result["plan"]["s2_5_patch_only_repair"]["selected_candidate_id"] == "missing_candidate"
    blocked = json.loads((paths.root / "plan" / "s2_5_patch_only_repair.json").read_text(encoding="utf-8"))
    assert blocked["implementation_blocked"] is True
    manifest = json.loads((paths.root / "plan" / "code_patches" / "patch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "no_valid_patch"
    assert manifest["implementation_blocked"] is True
    assert (paths.root / "plan" / "s2_planner" / "planner_gate_report.json").exists()
    patch_gate = json.loads((paths.root / "plan" / "code_patches" / "patch_gate_report.json").read_text(encoding="utf-8"))
    assert patch_gate["gate"] == "fail"
    assert patch_gate["repairable"] is True


def test_c2c_s2_variant_scorer_prefers_diverse_failure_targeted_variant(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {"eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"], "gpu_ids": [0], "max_candidates": 1},
        "allowed_files": ["rosetta/model/projector.py", "rosetta/model/wrapper.py", "script/train/SFT_train.py"],
    }
    config["agents"] = {"s2_directional_planner": {"resume_enabled": False, "max_variant_candidates": 3}}
    config["code_patch"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_s2_variant_scorer", simulate=False)
    ideas = default_c2c_ideas("topic", config["c2c"]["baseline"])
    ArtifactManager(paths.root).write_json("S1_literature", "ideas.json", ideas, artifact_type="ideas", summary="ideas")
    ArtifactManager(paths.root).write_json("S1_literature", "c2c/baseline_evidence.json", config["c2c"]["baseline"], artifact_type="baseline", summary="baseline")
    write_json(
        paths.root / "plan" / "performance_feedback.json",
        {
            "summary": {
                "recommended_s2_action": "mechanism_repair",
                "repair_vs_variant_signals": ["single_dataset_small_drop"],
            },
            "candidate_results": [
                {
                    "id": "old_projector_variant",
                    "proxy_screen": {"proxy_dataset_deltas": {"mmlu-redux": 0.2, "ai2-arc": -0.8, "openbookqa": 0.1}},
                    "failure_attribution": {
                        "patch_risk": {
                            "risk_files": [{"path": "rosetta/model/projector.py"}],
                            "risk_labels": ["projector_mechanism_changed"],
                        }
                    },
                }
            ],
        },
    )
    write_json(
        paths.root / "plan" / "s2_planner_memory.json",
        {
            "schema_version": "c2c_s2_planner_memory_v1",
            "entries": [
                {
                    "selected_candidate": {
                        "id": "old_projector_variant",
                        "variant_fingerprint": "oldfp",
                        "mechanism_axis": "routing",
                        "integration_point": "projector",
                        "control_signal": "utility",
                    },
                    "candidate_summaries": [],
                    "feedback_digest": {"patch_risk_files": ["rosetta/model/projector.py"]},
                }
            ],
        },
    )

    class PlannerLLM(ModelClient):
        def __init__(self, config, project_root=None):
            super().__init__(config, project_root=project_root)
            self.use_real_api = True

        def generate_json_with_schema(self, **kwargs):
            return {
                "planner_summary": "Generate projector repeat and wrapper repair variants.",
                "planning_mode": "same_direction_variant",
                "variant_candidates": [
                    {
                        "id": "repeat_projector_router",
                        "title": "Repeat projector router",
                        "mechanism_axis": "routing",
                        "integration_point": "projector",
                        "control_signal": "utility",
                        "expected_dataset_tradeoff": {"mmlu-redux": "up", "ai2-arc": "risk", "openbookqa": "up"},
                        "risk_budget": {"max_changed_files": 3, "forbidden_files": ["script/evaluation/*"]},
                        "anti_repeat": "Still mostly projector routing.",
                        "description": "Repeat projector routing.",
                        "hypothesis": "Projector router improves utility.",
                        "mechanism_type": "utility_predicted_cache_routing",
                        "experiment_contract": {"expected_files": ["rosetta/model/projector.py"]},
                        "implementation_plan": {"integration_points": ["rosetta/model/projector.py"]},
                    },
                    {
                        "id": "wrapper_arc_recovery_residual",
                        "title": "Wrapper ARC recovery residual",
                        "mechanism_axis": "normalization",
                        "integration_point": "wrapper",
                        "control_signal": "span_agreement",
                        "expected_dataset_tradeoff": {"mmlu-redux": "flat", "ai2-arc": "up", "openbookqa": "flat"},
                        "risk_budget": {"max_changed_files": 2, "forbidden_files": ["script/evaluation/*"]},
                        "anti_repeat": "Moves integration from projector to wrapper and targets AI2-ARC regression.",
                        "description": "Use wrapper residual normalization to preserve positive datasets while recovering AI2-ARC.",
                        "hypothesis": "Wrapper residual normalization reduces AI2-ARC regression without losing MMLU/OpenBook gains.",
                        "mechanism_type": "utility_predicted_cache_routing",
                        "experiment_contract": {"expected_files": ["rosetta/model/wrapper.py"]},
                        "implementation_plan": {"integration_points": ["rosetta/model/wrapper.py"]},
                    },
                ],
            }

    context = AgentContext(paths.root, config, ArtifactManager(paths.root), PlannerLLM(config, project_root=paths.root))
    result = PlanAgent(context).run()

    planned = result["plan"]["candidate_ideas"]
    assert len(planned) == 1
    assert planned[0]["id"] == "wrapper_arc_recovery_residual"
    assert planned[0]["s2_variant"]["integration_point"] == "wrapper"
    assert planned[0]["s2_variant"]["variant_score"]["score"] > 0
    variant_artifact = json.loads((paths.root / "plan" / "next_variant.json").read_text(encoding="utf-8"))
    assert variant_artifact["next_variant"]["id"] == "wrapper_arc_recovery_residual"
    assert "wrapper_arc_recovery_residual" in variant_artifact["considered_variant_ids"]
    assert "repeat_projector_router" in variant_artifact["considered_variant_ids"]
    assert "targets_dragging_dataset" in variant_artifact["next_variant"]["variant_score"]["reasons"]


def test_c2c_s2_variant_scorer_penalizes_repeated_failed_integration_point_and_proxy_calibration(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {"eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"], "gpu_ids": [0], "max_candidates": 1},
        "allowed_files": ["rosetta/model/projector.py", "rosetta/model/wrapper.py"],
    }
    config["agents"] = {"s2_directional_planner": {"resume_enabled": False, "max_variant_candidates": 3}}
    config["code_patch"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_s2_calibration_penalty", simulate=False)
    ideas = default_c2c_ideas("topic", config["c2c"]["baseline"])
    ArtifactManager(paths.root).write_json("S1_literature", "ideas.json", ideas, artifact_type="ideas", summary="ideas")
    ArtifactManager(paths.root).write_json("S1_literature", "c2c/baseline_evidence.json", config["c2c"]["baseline"], artifact_type="baseline", summary="baseline")
    write_json(
        paths.root / "plan" / "performance_feedback.json",
        {
            "proxy_calibration": {
                "summary": {
                    "method_feedback": {
                        "risky_datasets": [{"dataset": "mmlu-redux", "misprediction_rate": 1.0}],
                        "risky_mechanisms": [{"mechanism_type": "utility_predicted_cache_routing", "false_positive_rate": 1.0}],
                        "risky_integration_points": [{"integration_point": "projector", "false_positive_rate": 1.0}],
                    }
                }
            }
        },
    )
    write_json(
        paths.root / "plan" / "s2_planner_memory.json",
        {
            "schema_version": "c2c_s2_planner_memory_v1",
            "entries": [
                {
                    "selected_candidate": {
                        "id": "failed_projector_1",
                        "integration_point": "projector",
                        "experiment_contract": {"expected_files": ["rosetta/model/projector.py"]},
                    },
                    "feedback_digest": {"latest_decision": "proxy_repairable"},
                },
                {
                    "selected_candidate": {
                        "id": "failed_projector_2",
                        "integration_point": "projector",
                        "experiment_contract": {"expected_files": ["rosetta/model/projector.py"]},
                    },
                    "feedback_digest": {"latest_decision": "not_viable"},
                },
            ],
        },
    )

    class PlannerLLM(ModelClient):
        def __init__(self, config, project_root=None):
            super().__init__(config, project_root=project_root)
            self.use_real_api = True

        def generate_json_with_schema(self, **kwargs):
            return {
                "planner_summary": "Compare failed projector reuse with wrapper alternative.",
                "planning_mode": "same_direction_variant",
                "variant_candidates": [
                    {
                        "id": "projector_proxy_overtrust",
                        "title": "Projector proxy overtrust",
                        "mechanism_axis": "routing",
                        "integration_point": "projector",
                        "control_signal": "utility",
                        "expected_dataset_tradeoff": {"mmlu-redux": "up", "ai2-arc": "flat", "openbookqa": "flat"},
                        "risk_budget": {"max_changed_files": 2},
                        "description": "Projector utility routing.",
                        "hypothesis": "Projector utility routing improves proxy.",
                        "mechanism_type": "utility_predicted_cache_routing",
                        "experiment_contract": {"expected_files": ["rosetta/model/projector.py"]},
                    },
                    {
                        "id": "wrapper_calibrated_variant",
                        "title": "Wrapper calibrated variant",
                        "mechanism_axis": "normalization",
                        "integration_point": "wrapper",
                        "control_signal": "span_agreement",
                        "expected_dataset_tradeoff": {"mmlu-redux": "flat", "ai2-arc": "up", "openbookqa": "flat"},
                        "risk_budget": {"max_changed_files": 2},
                        "description": "Wrapper normalization addresses calibration risk.",
                        "hypothesis": "Wrapper normalization avoids projector false positives.",
                        "mechanism_type": "semantic_span_graph_alignment",
                        "experiment_contract": {"expected_files": ["rosetta/model/wrapper.py"]},
                    },
                ],
            }

    context = AgentContext(paths.root, config, ArtifactManager(paths.root), PlannerLLM(config, project_root=paths.root))
    result = PlanAgent(context).run()
    planned = result["plan"]["candidate_ideas"]
    artifact = json.loads((paths.root / "plan" / "next_variant.json").read_text(encoding="utf-8"))

    assert planned[0]["id"] == "wrapper_calibrated_variant"
    assert artifact["next_variant"]["id"] == "wrapper_calibrated_variant"
    assert "projector_proxy_overtrust" in artifact["considered_variant_ids"]


def test_c2c_s2_resume_planner_uses_codex_session(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["codex_cli"] = {"use_resume": True, "sandbox": "read-only", "approval_policy": "never", "json_events": True}
    config["llm"].update({"model": "gpt-5.6-terra", "reasoning_effort": "xhigh"})
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {"eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"], "gpu_ids": [0], "max_candidates": 2},
        "allowed_files": ["rosetta/model/projector.py", "rosetta/model/wrapper.py", "script/train/SFT_train.py", "test/test_aligner_span_overlap.py"],
    }
    config["code_patch"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_s2_resume_planner", simulate=False)
    ideas = default_c2c_ideas("topic", config["c2c"]["baseline"])
    ArtifactManager(paths.root).write_json("S1_literature", "ideas.json", ideas, artifact_type="ideas", summary="ideas")
    ArtifactManager(paths.root).write_json("S1_literature", "c2c/baseline_evidence.json", config["c2c"]["baseline"], artifact_type="baseline", summary="baseline")

    monkeypatch.setattr("auto_research.agents.plan.shutil.which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    commands = []
    prompts = []

    def fake_run(command, **kwargs):
        commands.append(command)
        prompts.append(kwargs.get("input") or "")
        assert command[0] == "codex"
        assert "-s" in command and command[command.index("-s") + 1] == "read-only"
        assert "--json" in command
        assert command[command.index("-m") + 1] == "gpt-5.6-terra"
        assert 'model_reasoning_effort="xhigh"' in command
        assert command[-1] == "-"
        output_path = Path(command[command.index("--output-last-message") + 1])
        variant_id = "utility_resume_memory_variant" if "resume" in command else "utility_resume_soft_residual"
        payload = {
            "planner_summary": "Resume planner inspected memory and made a soft residual variant.",
            "planning_mode": "same_direction_variant",
            "candidates": [
                {
                    "id": variant_id,
                    "title": "Utility resume soft residual",
                    "description": "Use utility prediction to softly scale residual transfer while preserving baseline cache coverage.",
                    "motivation": "The previous direction collapsed all datasets, so preserve coverage and only modulate residual transfer.",
                    "hypothesis": "Soft residual utility routing avoids all-dataset collapse in cheap proxy.",
                    "mechanism_type": "utility_predicted_cache_routing",
                    "paper_claim": "Receiver utility should modulate residual cache transfer rather than hard-filtering spans.",
                    "why_baseline_fails": "The baseline lacks a downstream utility signal for residual transfer.",
                    "expected_signature": {
                        "primary": "residual scale changes without coverage collapse",
                        "stats": ["utility_residual_scale", "baseline_transfer_coverage"],
                    },
                    "experiment_contract": {
                        "config_overrides": {
                            "train": {"model": {"cache_routing_mode": variant_id}},
                            "eval": {"model": {"rosetta_config": {"cache_routing_mode": variant_id}}},
                        }
                    },
                    "failure_avoidance": ["preserve baseline coverage"],
                    "failure_feedback_refs": [{"source_type": "failure_feedback", "source_label": "memory"}],
                }
            ],
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614174111"}\n',
            stderr="",
        )

    import auto_research.agents.plan as plan_module

    monkeypatch.setattr(plan_module.subprocess, "run", fake_run)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))

    result = PlanAgent(context).run()

    assert result["plan"]["directional_planning"]["source"] == "codex_resume_planner"
    assert result["plan"]["directional_planning"]["session_id"] == "123e4567-e89b-12d3-a456-426614174111"
    assert result["plan"]["candidate_ideas"][0]["id"] == "utility_resume_soft_residual"
    sessions = yaml.safe_load((paths.root / "meta" / "codex_sessions.yaml").read_text(encoding="utf-8"))
    assert sessions["sessions"]["s2_planner:utility_predicted_cache_routing"]["session_id"] == "123e4567-e89b-12d3-a456-426614174111"
    events = (paths.root / "plan" / "logs" / "s2_planner_codex_events.jsonl").read_text(encoding="utf-8")
    assert "s2_planner:utility_predicted_cache_routing" in events
    assert "resume" not in commands[0]

    second = PlanAgent(context).run()

    assert second["plan"]["directional_planning"]["used_existing_session"] is True
    assert second["plan"]["candidate_ideas"][0]["id"] == "utility_resume_memory_variant"
    assert "resume" in commands[1]
    assert commands[1][commands[1].index("resume") + 1] == "123e4567-e89b-12d3-a456-426614174111"
    assert "utility_resume_soft_residual" in prompts[1]
    memory = json.loads((paths.root / "plan" / "s2_planner_memory.json").read_text(encoding="utf-8"))
    assert memory["entry_count"] == 2


def test_c2c_s2_resume_planner_inherits_mechanism_type_from_s1_direction(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["codex_cli"] = {"use_resume": True, "sandbox": "read-only", "approval_policy": "never", "json_events": True}
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {"eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"], "gpu_ids": [0], "max_candidates": 2},
        "allowed_files": ["rosetta/model/aligner.py", "rosetta/model/projector.py", "rosetta/model/wrapper.py"],
    }
    config["code_patch"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_s2_resume_inherit_mechanism", simulate=False)
    s1_direction = {
        "id": "pathology_conditioned_transfer_controller",
        "title": "Pathology-Conditioned Transfer Controller",
        "selected": True,
        "hypothesis": "Condition transfer on alignment pathology buckets.",
        "description": "Use pathology statistics to abstain or attenuate harmful transfer.",
        "mechanism_type": "pathology_conditioned_controller",
        "expected_files": ["rosetta/model/aligner.py", "rosetta/model/projector.py", "rosetta/model/wrapper.py"],
    }
    ArtifactManager(paths.root).write_json("S1_literature", "ideas.json", [s1_direction], artifact_type="ideas", summary="ideas")
    ArtifactManager(paths.root).write_json("S1_literature", "c2c/baseline_evidence.json", config["c2c"]["baseline"], artifact_type="baseline", summary="baseline")
    monkeypatch.setattr("auto_research.agents.plan.shutil.which", lambda name: "/usr/bin/codex" if name == "codex" else None)

    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "planner_summary": "Make a bucketed abstain variant.",
                    "planning_mode": "new_direction_after_budget",
                    "candidates": [
                        {
                            "id": "pathology_conditioned_transfer_controller_bucketed_abstain_v1",
                            "title": "Bucketed pathology abstain",
                            "description": "Abstain only on high-pathology alignment buckets.",
                            "hypothesis": "High-pathology abstention removes localized harmful transfer.",
                            "mechanism_type": "pathology_conditioned_transfer_controller",
                            "experiment_contract": {},
                            "implementation_plan": {},
                            "failure_avoidance": ["no evaluator changes"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614174333"}\n',
            stderr="",
        )

    import auto_research.agents.plan as plan_module

    monkeypatch.setattr(plan_module.subprocess, "run", fake_run)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))

    result = PlanAgent(context).run()

    planned = result["plan"]["candidate_ideas"]
    assert result["plan"]["directional_planning"]["status"] == "ok"
    assert planned[0]["id"] == "pathology_conditioned_transfer_controller_bucketed_abstain_v1"
    assert planned[0]["mechanism_type"] == "pathology_conditioned_controller"
    assert planned[0]["novelty_gate"]["status"] == "pass"
    assert planned[0]["implementation_scope_gate"]["status"] == "pass"
    assert planned[0]["experiment_contract"]["config_overrides"]["train"]["model"]["cache_controller_mode"] == "pathology_conditioned_transfer_controller_bucketed_abstain_v1"


def test_c2c_s2_resume_planner_resets_duplicate_session_but_keeps_memory(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["codex_cli"] = {"use_resume": True, "sandbox": "read-only", "approval_policy": "never", "json_events": True}
    config["agents"] = {"s2_directional_planner": {"session_reset_duplicate_streak": 1}}
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {"eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"], "gpu_ids": [0], "max_candidates": 2},
        "allowed_files": ["rosetta/model/projector.py", "rosetta/model/wrapper.py", "script/train/SFT_train.py", "test/test_aligner_span_overlap.py"],
    }
    config["code_patch"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_s2_resume_reset", simulate=False)
    ideas = default_c2c_ideas("topic", config["c2c"]["baseline"])
    ArtifactManager(paths.root).write_json("S1_literature", "ideas.json", ideas, artifact_type="ideas", summary="ideas")
    ArtifactManager(paths.root).write_json("S1_literature", "c2c/baseline_evidence.json", config["c2c"]["baseline"], artifact_type="baseline", summary="baseline")

    monkeypatch.setattr("auto_research.agents.plan.shutil.which", lambda name: "/usr/bin/codex" if name == "codex" else None)

    def payload_for_variant(variant_id: str) -> dict:
        return {
            "planner_summary": "Duplicate reset test.",
            "planning_mode": "same_direction_variant",
            "candidates": [
                {
                    "id": variant_id,
                    "title": "Utility duplicate candidate",
                    "description": "Use utility prediction to softly scale residual transfer while preserving baseline cache coverage.",
                    "hypothesis": "Soft residual utility routing avoids all-dataset collapse in cheap proxy.",
                    "mechanism_type": "utility_predicted_cache_routing",
                    "paper_claim": "Receiver utility should modulate residual cache transfer rather than hard-filtering spans.",
                    "why_baseline_fails": "The baseline lacks a downstream utility signal for residual transfer.",
                    "expected_signature": {"primary": "residual scale changes without coverage collapse", "stats": ["utility_residual_scale"]},
                    "experiment_contract": {
                        "config_overrides": {
                            "train": {"model": {"cache_routing_mode": variant_id}},
                            "eval": {"model": {"rosetta_config": {"cache_routing_mode": variant_id}}},
                        }
                    },
                    "failure_avoidance": ["preserve baseline coverage"],
                }
            ],
        }

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(json.dumps(payload_for_variant("utility_duplicate_candidate")), encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614174222"}\n',
            stderr="",
        )

    import auto_research.agents.plan as plan_module

    monkeypatch.setattr(plan_module.subprocess, "run", fake_run)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))

    first = PlanAgent(context).run()
    second = PlanAgent(context).run()

    assert first["plan"]["directional_planning"]["source"] == "codex_resume_planner"
    assert first["plan"]["candidate_ideas"][0]["id"] == "utility_duplicate_candidate"
    assert second["plan"]["directional_planning"]["status"] == "fallback_no_real_llm"
    assert second["plan"]["directional_planning"]["resume_planner"]["session_reset"] is True
    assert second["plan"]["directional_planning"]["resume_planner"]["session_reset_reason"] == "duplicate_output_streak"
    assert "resume" in commands[1]
    sessions = yaml.safe_load((paths.root / "meta" / "codex_sessions.yaml").read_text(encoding="utf-8"))
    assert "s2_planner:utility_predicted_cache_routing" not in sessions.get("sessions", {})
    memory = json.loads((paths.root / "plan" / "s2_planner_memory.json").read_text(encoding="utf-8"))
    assert memory["entry_count"] == 2
    assert memory["entries"][0]["selected_candidate"]["id"] == "utility_duplicate_candidate"
    events = (paths.root / "plan" / "logs" / "s2_planner_codex_events.jsonl").read_text(encoding="utf-8")
    assert "session_reset" in events


def test_c2c_s2_resume_planner_real_api_skips_gpt_fallback_after_duplicate_reset(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["use_real_api"] = True
    config["llm"]["codex_cli"] = {"use_resume": True, "sandbox": "read-only", "approval_policy": "never", "json_events": True}
    config["agents"] = {"s2_directional_planner": {"session_reset_duplicate_streak": 1}}
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {"eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"], "gpu_ids": [0], "max_candidates": 2},
        "allowed_files": ["rosetta/model/projector.py", "rosetta/model/wrapper.py", "script/train/SFT_train.py", "test/test_aligner_span_overlap.py"],
    }
    config["code_patch"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_s2_resume_real_api_no_gpt_fallback", simulate=False)
    ideas = default_c2c_ideas("topic", config["c2c"]["baseline"])
    artifacts = ArtifactManager(paths.root)
    artifacts.write_json("S1_literature", "ideas.json", ideas, artifact_type="ideas", summary="ideas")
    artifacts.write_json("S1_literature", "c2c/baseline_evidence.json", config["c2c"]["baseline"], artifact_type="baseline", summary="baseline")

    monkeypatch.setattr("auto_research.agents.plan.shutil.which", lambda name: "/usr/bin/codex" if name == "codex" else None)

    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "planner_summary": "Duplicate reset test.",
                    "planning_mode": "same_direction_variant",
                    "candidates": [
                        {
                            "id": "utility_duplicate_candidate",
                            "title": "Utility duplicate candidate",
                            "description": "Use utility prediction to softly scale residual transfer while preserving baseline cache coverage.",
                            "hypothesis": "Soft residual utility routing avoids all-dataset collapse in cheap proxy.",
                            "mechanism_type": "utility_predicted_cache_routing",
                            "experiment_contract": {
                                "config_overrides": {
                                    "train": {"model": {"cache_routing_mode": "utility_duplicate_candidate"}},
                                    "eval": {"model": {"rosetta_config": {"cache_routing_mode": "utility_duplicate_candidate"}}},
                                }
                            },
                            "failure_avoidance": ["preserve baseline coverage"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614174333"}\n',
            stderr="",
        )

    import auto_research.agents.plan as plan_module

    monkeypatch.setattr(plan_module.subprocess, "run", fake_run)
    client = ModelClient(config, project_root=paths.root)
    client.use_real_api = True

    def fail_gpt_fallback(**_kwargs):
        raise AssertionError("S2 should not call GPT fallback after Codex resume duplicate reset in real API mode")

    client.generate_json_with_schema = fail_gpt_fallback
    context = AgentContext(paths.root, config, artifacts, client)

    first = PlanAgent(context).run()
    second = PlanAgent(context).run()

    assert first["plan"]["directional_planning"]["source"] == "codex_resume_planner"
    assert second["plan"]["directional_planning"]["status"] == "fallback_resume_planner_unavailable"
    assert second["plan"]["directional_planning"]["resume_planner"]["session_reset"] is True
    assert second["plan"]["candidate_ideas"][0]["s2_planner"]["source"] == "fallback_s1_ideas"


def test_c2c_novelty_report_rejects_pure_local_tuning() -> None:
    report = c2c_idea_novelty_report(
        {
            "id": "local_topk_tuning",
            "title": "Local top-k confidence floor tuning",
            "selected": True,
            "experiment_contract": {
                "primary_metric": "three_dataset_mean",
                "baseline": "base",
                "config_overrides": {
                    "train": {"model": {"soft_alignment_top_k": 2, "soft_alignment_confidence_floor": 0.2}},
                    "eval": {"model": {"rosetta_config": {"soft_alignment_top_k": 2}}},
                },
            },
        }
    )

    assert report["status"] == "reject"
    assert report["local_tuning_flags"]


def test_c2c_novelty_report_rejects_hard_gate_without_coverage_controls() -> None:
    report = c2c_idea_novelty_report(
        {
            "id": "hard_gate_stack",
            "title": "Utility hard gate stack",
            "description": "Add an additional hard gate that rejects transferred spans whenever utility is below a fixed threshold.",
            "mechanism_type": "utility_predicted_cache_routing",
            "paper_claim": "Cache routing should use utility.",
            "why_baseline_fails": "The baseline accepts harmful spans.",
            "expected_signature": {"primary": "fewer bad spans"},
            "ablation_plan": {"switch": "ablation_disable_hard_gate_stack"},
            "expected_files": ["rosetta/model/projector.py"],
            "experiment_contract": {"ablation_switch": "ablation_disable_hard_gate_stack"},
        }
    )

    assert report["status"] == "reject"
    assert report["hard_gate_stack_flags"]
    assert set(report["missing_required_fields"]) == {"coverage_diagnostics", "matched_coverage_ablation"}


def test_c2c_novelty_report_accepts_default_mechanism_idea() -> None:
    idea = default_c2c_ideas("topic", {"name": "base", "mean": 50.0, "datasets": {}})[0]
    report = c2c_idea_novelty_report(idea)

    assert report["status"] == "pass"
    assert report["mechanism_type"] == "utility_predicted_cache_routing"
    assert "coverage_diagnostics" in report["signals"]
    assert "matched_coverage_ablation" in report["signals"]


def test_s2_gate_tracks_missing_coverage_controls_as_debt_by_default(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "plan").mkdir(parents=True)
    (project / "plan" / "short_loop_plan.yaml").write_text("collector: c2c_small_loop\n", encoding="utf-8")
    ideas = default_c2c_ideas("topic", {"name": "base", "mean": 50.0, "datasets": {}})
    (project / "plan" / "candidate_ideas.json").write_text(json.dumps(ideas), encoding="utf-8")
    plan = {
        "selected_idea": ideas[0],
        "candidate_ideas": ideas,
        "hypotheses": [{"id": "h1"}],
        "baselines": [{"name": "base"}, {"name": "candidate"}],
        "datasets": [{"name": "mmlu-redux"}],
        "metrics": [],
        "statistical_testing": {},
        "ablation_matrix": [],
        "task_graph": {},
        "resource_budget": {},
        "execution": {
            "collector": "c2c_small_loop",
            "min_delta_to_pass": 0.1,
            "max_dataset_regression": 2.0,
        },
        "acceptance_criteria": {
            "minimum_mean_delta": 0.1,
            "max_dataset_regression": 2.0,
        },
        "reviewer_risk_controls": {"top_concerns": []},
    }
    (project / "plan" / "plan.yaml").write_text(yaml.safe_dump(plan), encoding="utf-8")
    _write_direction_and_variant_gate_artifacts(project)

    report = S2GateValidator(project, {}).validate()

    assert report.status == "PASS"
    check_names = {check.name for check in report.checks if check.status == "PASS"}
    assert "c2c_coverage_control_requirements" in check_names
    assert "c2c_matched_coverage_ablation" in check_names
    debt_checks = [check for check in report.checks if check.name == "c2c_coverage_control_requirements"]
    assert debt_checks[0].details["quality_debt"] is True


def test_s2_gate_requires_coverage_controls_in_strict_mode(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "plan").mkdir(parents=True)
    (project / "plan" / "short_loop_plan.yaml").write_text("collector: c2c_small_loop\n", encoding="utf-8")
    ideas = default_c2c_ideas("topic", {"name": "base", "mean": 50.0, "datasets": {}})
    (project / "plan" / "candidate_ideas.json").write_text(json.dumps(ideas), encoding="utf-8")
    plan = {
        "selected_idea": ideas[0],
        "candidate_ideas": ideas,
        "hypotheses": [{"id": "h1"}],
        "baselines": [{"name": "base"}, {"name": "candidate"}],
        "datasets": [{"name": "mmlu-redux"}],
        "metrics": [],
        "statistical_testing": {},
        "ablation_matrix": [],
        "task_graph": {},
        "resource_budget": {},
        "execution": {
            "collector": "c2c_small_loop",
            "min_delta_to_pass": 0.1,
            "max_dataset_regression": 2.0,
        },
        "acceptance_criteria": {
            "minimum_mean_delta": 0.1,
            "max_dataset_regression": 2.0,
        },
        "reviewer_risk_controls": {"top_concerns": []},
    }
    (project / "plan" / "plan.yaml").write_text(yaml.safe_dump(plan), encoding="utf-8")
    _write_direction_and_variant_gate_artifacts(project)

    report = S2GateValidator(project, {"code_patch": {"validation": {"gate_mode": "strict"}}}).validate()

    assert report.status == "NEEDS_RETRY"
    check_names = {check.name for check in report.checks if check.status == "NEEDS_RETRY"}
    assert "c2c_coverage_control_requirements" in check_names
    assert "c2c_matched_coverage_ablation" in check_names


def test_c2c_debate_structures_fallback_refs(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=True)
    config["ideation"] = {"debate": {"enabled": True, "rounds": 1}}
    context = AgentContext(tmp_path / "workspace" / "p", config, ArtifactManager(tmp_path / "workspace" / "p"), ModelClient(config, project_root=tmp_path))

    debate = MultiAgentReasoningService(context).run_c2c_debate(
        topic="cross tokenizer cache",
        repo_card={},
        paper_cards=[],
        rebuttal_matrix={"top_concerns": ["failure_modes_ood"]},
        negative_memory={},
        baseline={"name": "base", "mean": 50.0, "datasets": {}},
        feedback=[],
    )

    fallback_idea = debate["selected_ideas"][0]
    assert isinstance(fallback_idea["evidence_refs"][0], dict)
    assert fallback_idea["evidence_refs"][0]["source_path"]
    assert fallback_idea["counterevidence_refs"][0]["source_type"] in {"repo_artifact", "summary"}
    assert fallback_idea["code_refs"][0]["source_type"] == "code"


def test_c2c_debate_parallel_timeout_falls_back(tmp_path: Path, monkeypatch) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["use_real_api"] = True
    config["llm"]["timeout_seconds"] = 20
    config["ideation"] = {"debate": {"enabled": True, "rounds": 1, "parallel": True, "agent_timeout_seconds": 1}}
    context = AgentContext(tmp_path / "workspace" / "p", config, ArtifactManager(tmp_path / "workspace" / "p"), ModelClient(config, project_root=tmp_path))
    context.llm.use_real_api = True

    def slow_worker(*args, **kwargs):
        time.sleep(30)

    monkeypatch.setattr("auto_research.agents.debate._run_role_worker", slow_worker)

    debate = MultiAgentReasoningService(context).run_c2c_debate(
        topic="cross tokenizer cache",
        repo_card={},
        paper_cards=[],
        rebuttal_matrix={"top_concerns": ["failure_modes_ood"]},
        negative_memory={},
        baseline={"name": "base", "mean": 50.0, "datasets": {}},
        feedback=[],
    )

    statuses = [output.get("status") for output in debate["rounds"][0]["outputs"]]
    assert statuses == ["timeout_fallback"] * 6
    assert debate["selected_ideas"]
    progress_path = tmp_path / "workspace" / "p" / "literature" / "c2c" / "idea_debate_progress.jsonl"
    assert progress_path.exists()
    assert "timeout_fallback" in progress_path.read_text(encoding="utf-8")


def test_c2c_debate_role_specific_timeout_and_recovery_flag(tmp_path: Path, monkeypatch) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["use_real_api"] = True
    config["ideation"] = {
        "debate": {
            "enabled": True,
            "rounds": 2,
            "parallel": True,
            "agent_timeout_seconds": 1,
            "role_timeout_seconds": {"method_inventor": 2},
        }
    }
    context = AgentContext(tmp_path / "workspace" / "p", config, ArtifactManager(tmp_path / "workspace" / "p"), ModelClient(config, project_root=tmp_path))
    context.llm.use_real_api = True

    def role_worker(queue, config, project_root_text, role, context_payload, prior_round, round_idx, fallback):
        if role == "method_inventor" and round_idx == 1:
            time.sleep(1.4)
        if role == "systems_feasibility" and round_idx == 1:
            time.sleep(3)
        output = dict(fallback)
        output["status"] = "ok"
        queue.put({"status": "ok", "output": output})

    monkeypatch.setattr("auto_research.agents.debate._run_role_worker", role_worker)

    debate = MultiAgentReasoningService(context).run_c2c_debate(
        topic="cross tokenizer cache",
        repo_card={},
        paper_cards=[],
        rebuttal_matrix={"top_concerns": ["failure_modes_ood"]},
        negative_memory={},
        baseline={"name": "base", "mean": 50.0, "datasets": {}},
        feedback=[],
    )

    assert debate["rounds"][0]["outputs"][2]["role"] == "method_inventor"
    assert debate["rounds"][0]["outputs"][2]["status"] == "ok"
    assert debate["rounds"][0]["outputs"][4]["role"] == "systems_feasibility"
    assert debate["rounds"][0]["outputs"][4]["status"] == "timeout_fallback"
    assert debate["rounds"][1]["outputs"][4]["status"] == "ok"
    assert any(
        flag.get("type") == "gpt_recovered_after_timeout" and flag.get("role") == "systems_feasibility"
        for flag in debate["quality_flags"]
    )
    progress_path = tmp_path / "workspace" / "p" / "literature" / "c2c" / "idea_debate_progress.jsonl"
    progress = progress_path.read_text(encoding="utf-8")
    assert '"role": "method_inventor"' in progress
    assert '"timeout_seconds": 2' in progress
    assert "systems_feasibility timed out after 1s" in progress


def test_c2c_debate_meta_timeout_falls_back(tmp_path: Path, monkeypatch) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["use_real_api"] = True
    config["ideation"] = {
        "debate": {
            "enabled": True,
            "rounds": 1,
            "parallel": True,
            "agent_timeout_seconds": 5,
            "meta_timeout_seconds": 1,
        }
    }
    context = AgentContext(tmp_path / "workspace" / "p", config, ArtifactManager(tmp_path / "workspace" / "p"), ModelClient(config, project_root=tmp_path))
    context.llm.use_real_api = True

    def fast_role_worker(queue, config, project_root_text, role, context_payload, prior_round, round_idx, fallback):
        queue.put({"status": "ok", "output": fallback})

    def slow_meta_worker(*args, **kwargs):
        time.sleep(30)

    monkeypatch.setattr("auto_research.agents.debate._run_role_worker", fast_role_worker)
    monkeypatch.setattr("auto_research.agents.debate._run_meta_worker", slow_meta_worker)

    debate = MultiAgentReasoningService(context).run_c2c_debate(
        topic="cross tokenizer cache",
        repo_card={},
        paper_cards=[],
        rebuttal_matrix={"top_concerns": ["failure_modes_ood"]},
        negative_memory={},
        baseline={"name": "base", "mean": 50.0, "datasets": {}},
        feedback=[],
    )

    assert debate["meta_judge"]["status"] == "timeout_fallback"
    assert any(flag.get("type") == "meta_timeout_fallback" for flag in debate["quality_flags"])
    assert debate["selected_ideas"]
    progress_path = tmp_path / "workspace" / "p" / "literature" / "c2c" / "idea_debate_progress.jsonl"
    assert "meta_judge" in progress_path.read_text(encoding="utf-8")


def test_c2c_debate_meta_receives_compressed_round_summaries(tmp_path: Path, monkeypatch) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["use_real_api"] = True
    config["ideation"] = {
        "debate": {
            "enabled": True,
            "rounds": 2,
            "parallel": False,
            "agent_timeout_seconds": 5,
            "meta_timeout_seconds": 5,
        }
    }
    context = AgentContext(tmp_path / "workspace" / "p", config, ArtifactManager(tmp_path / "workspace" / "p"), ModelClient(config, project_root=tmp_path))
    context.llm.use_real_api = True

    def fake_role(role, context_payload, prior_round, round_idx):
        return {
            "role": role,
            "status": "ok",
            "score": 7,
            "claims": [f"{role} claim"],
            "evidence_refs": [{"source_type": "paper", "source_label": "paper"}],
            "counterevidence_refs": [{"source_type": "failure_feedback", "source_label": "feedback"}],
            "code_refs": [{"source_type": "code", "source_label": "code"}],
            "failure_feedback_refs": [{"source_type": "failure_feedback", "source_label": "feedback"}],
            "proposed_ideas": [{"id": "idea_a", "title": "Idea A", "selected": role == "literature_scout", "novelty_score": 7, "feasibility_score": 8}],
            "decision_chain": {"evidence": ["e1"], "counterevidence": ["c1"], "conclusion": "ok"},
            "risks": ["r1"],
        }

    captured = {}

    def fake_meta(context_payload, round_summaries, fallback_ideas, fallback):
        captured["round_summaries"] = round_summaries
        return fallback

    monkeypatch.setattr("auto_research.agents.debate.MultiAgentReasoningService._run_role", lambda self, role, context_payload, prior_round, round_idx: fake_role(role, context_payload, prior_round, round_idx))
    monkeypatch.setattr("auto_research.agents.debate.MultiAgentReasoningService._run_meta_judge_sync", lambda self, context_payload, round_summaries, fallback_ideas, fallback: fake_meta(context_payload, round_summaries, fallback_ideas, fallback))

    debate = MultiAgentReasoningService(context).run_c2c_debate(
        topic="cross tokenizer cache",
        repo_card={},
        paper_cards=[],
        rebuttal_matrix={"top_concerns": ["failure_modes_ood"]},
        negative_memory={},
        baseline={"name": "base", "mean": 50.0, "datasets": {}},
        feedback=[],
    )

    assert captured["round_summaries"]
    assert captured["round_summaries"][0]["role_summaries"][0]["top_evidence"]
    assert captured["round_summaries"][0]["selected_idea_ids"] == ["idea_a"]
    assert debate["selected_ideas"]


def test_c2c_feedback_bundle_expands_round_file(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    feedback_dir = project_root / "literature" / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    round_payload = {
        "created_at": "2026-05-19T00:00:00Z",
        "project_id": "proj_x",
        "iteration": 2,
        "kind": "c2c_feedback_summary",
        "summary_entry": {
            "timestamp": "2026-05-19T00:00:00Z",
            "project_id": "proj_x",
            "iteration": 2,
            "kind": "c2c_feedback_summary",
            "failed_idea_ids": ["idea_a"],
            "failed_titles": ["Idea A"],
            "avoid_repeat_rules": ["Avoid A"],
            "summary_text": "latest=c2c_failure_feedback:not_viable",
        },
        "entries": [
            {
                "timestamp": "2026-05-19T00:00:00Z",
                "project_id": "proj_x",
                "iteration": 2,
                "kind": "c2c_failure_feedback",
                "idea_id": "idea_a",
                "title": "Idea A",
                "decision": "not_viable",
                "failure_mode": "not_viable",
                "reason": "bad",
                "avoid_repeat_rule": "Avoid A",
            }
        ],
        "iteration_traces": [
            {
                "timestamp": "2026-05-19T00:00:01Z",
                "from_stage": "S3_experiment",
                "to_stage": "S1_literature",
                "iteration": 2,
                "reason": "bad",
                "result_status": "not_viable",
            }
        ],
        "feedback_items": [
            {
                "timestamp": "2026-05-19T00:00:00Z",
                "project_id": "proj_x",
                "iteration": 2,
                "kind": "c2c_feedback_summary",
                "failed_idea_ids": ["idea_a"],
                "failed_titles": ["Idea A"],
                "avoid_repeat_rules": ["Avoid A"],
            }
        ],
    }
    (feedback_dir / "failed_ideas_round_002.json").write_text(json.dumps(round_payload), encoding="utf-8")

    bundle = load_c2c_feedback_bundle(project_root)
    assert bundle["summary"]["failed_idea_ids"] == ["idea_a"]
    assert bundle["entries"]
    assert bundle["iteration_traces"]
    assert any(item.get("kind") == "c2c_failure_feedback" for item in bundle["feedback_items"])


def test_c2c_feedback_bundle_includes_direction_scorecard_method_view(tmp_path: Path) -> None:
    project_root = tmp_path / "project_direction_scorecard"
    path = project_root / "plan" / "direction_scorecard.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
            "schema_version": "c2c_direction_scorecard_v1",
            "project_id": "project_direction_scorecard",
            "current_direction_id": "utility_predicted_cache_routing",
            "current_direction": {
                "direction_id": "utility_predicted_cache_routing",
                "title": "Utility-predicted cache routing",
                "mechanism_type": "utility_predicted_cache_routing",
                "summary": {
                    "status": "budget_exhausted",
                    "attempt_count": 5,
                    "same_direction_failure_count": 5,
                    "same_direction_failure_budget": 5,
                    "best_proxy_delta": -1.2,
                    "positive_dataset_signal_attempts": 1,
                    "runtime_stable_attempts": 4,
                    "low_patch_risk_attempts": 4,
                    "all_dataset_collapse_attempts": 3,
                    "health_score": -8.4,
                    "direction_quality": "poor_direction_evidence",
                },
                "s1_feedback": {
                    "recommendation": "return_to_s1_new_direction",
                    "conclusion": "Direction failed after five attempts with repeated all-dataset collapse.",
                    "avoid_repeat_rule": "Do not repeat this S1 direction without a mechanism-level change.",
                },
            },
            }
        ),
        encoding="utf-8",
    )

    bundle = load_c2c_feedback_bundle(project_root, view="method")

    entries = bundle["entries"]
    scorecards = [item.get("direction_scorecard") for item in entries if item.get("direction_scorecard")]
    assert scorecards
    assert scorecards[0]["direction_id"] == "utility_predicted_cache_routing"
    assert scorecards[0]["summary"]["best_proxy_delta"] == -1.2
    assert scorecards[0]["s1_feedback"]["recommendation"] == "return_to_s1_new_direction"


def test_c2c_feedback_bundle_includes_proxy_calibration_method_view(tmp_path: Path) -> None:
    project_root = tmp_path / "project_proxy_calibration"
    path = project_root / "experiment" / "results" / "proxy_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "c2c_proxy_calibration_v1",
                "project_id": "project_proxy_calibration",
                "summary": {
                    "candidate_count": 2,
                    "proxy_false_positive_count": 1,
                    "proxy_false_positive_rate": 0.5,
                    "dataset_error_summary": {
                        "mmlu-redux": {
                            "mean_abs_proxy_full_delta_error": 2.4,
                            "max_abs_proxy_full_delta_error": 2.4,
                            "misprediction_count": 1,
                            "count": 1,
                        }
                    },
                    "mechanism_false_positive_summary": {
                        "utility_predicted_cache_routing": {
                            "count": 1,
                            "false_positive_count": 1,
                            "false_positive_rate": 1.0,
                        }
                    },
                },
                "current_iteration": {
                    "iteration": 3,
                    "acceptance_passed": False,
                    "candidate_count": 1,
                    "proxy_false_positive_count": 1,
                    "proxy_false_positive_rate": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = load_c2c_feedback_bundle(project_root, view="method")

    calibrations = [item.get("proxy_calibration") for item in bundle["entries"] if item.get("proxy_calibration")]
    assert calibrations
    assert calibrations[0]["summary"]["proxy_false_positive_rate"] == 0.5
    assert calibrations[0]["summary"]["dataset_error_summary"]["mmlu-redux"]["misprediction_count"] == 1
    assert calibrations[0]["summary"]["mechanism_false_positive_summary"]["utility_predicted_cache_routing"]["false_positive_rate"] == 1.0


def test_c2c_feedback_bundle_summary_builder() -> None:
    bundle = build_c2c_feedback_bundle(
        [
            {
                "timestamp": "2026-05-19T00:00:00Z",
                "project_id": "proj_x",
                "iteration": 2,
                "kind": "c2c_failure_feedback",
                "idea_id": "idea_a",
                "title": "Idea A",
                "decision": "not_viable",
                "failure_mode": "not_viable",
                "reason": "bad",
                "avoid_repeat_rule": "Avoid A",
            }
        ],
        project_id="proj_x",
        iteration=2,
        traces=[
            {
                "timestamp": "2026-05-19T00:00:01Z",
                "from_stage": "S3_experiment",
                "to_stage": "S1_literature",
                "iteration": 2,
                "reason": "bad",
                "result_status": "not_viable",
            }
        ],
        sources=["meta/negative_memory.jsonl"],
    )
    assert bundle["summary"]["failed_idea_ids"] == ["idea_a"]
    assert bundle["summary_entry"]["kind"] == "c2c_feedback_summary"
    assert bundle["feedback_items"][0]["kind"] == "c2c_feedback_summary"
    assert bundle["feedback_items"][1]["idea_id"] == "idea_a"


def test_c2c_feedback_bundle_preserves_failure_attribution() -> None:
    bundle = build_c2c_feedback_bundle(
        [
            {
                "timestamp": "2026-05-19T00:00:00Z",
                "project_id": "proj_x",
                "iteration": 2,
                "kind": "c2c_failure_feedback",
                "idea_id": "idea_a",
                "title": "Idea A",
                "decision": "not_viable",
                "failure_mode": "not_viable",
                "failure_attribution": {
                    "primary_failure": "mmlu-redux_regression",
                    "dragging_datasets": [
                        {"dataset": "mmlu-redux", "sample_family": "multi_domain_knowledge_reasoning", "regression": 3.2}
                    ],
                    "sample_type_failures": [
                        {"sample_family": "multi_domain_knowledge_reasoning", "dataset": "mmlu-redux"}
                    ],
                    "mixed_gain_patterns": ["openbookqa_gain_mmlu_redux_regression"],
                    "patch_risk": {
                        "risk_labels": ["projector_mechanism_changed"],
                        "risk_files": [{"path": "rosetta/model/projector.py", "reasons": ["projector mechanism changed"]}],
                    },
                },
            }
        ],
        project_id="proj_x",
        iteration=2,
    )

    summary = bundle["summary"]
    assert summary["dragging_datasets"][0]["dataset"] == "mmlu-redux"
    assert summary["sample_type_failures"] == ["multi_domain_knowledge_reasoning"]
    assert summary["patch_risk_files"] == ["rosetta/model/projector.py"]
    assert summary["mixed_gain_patterns"] == ["openbookqa_gain_mmlu_redux_regression"]
    assert "dragging_datasets=mmlu-redux" in summary["summary_text"]


def test_c2c_train_failure_with_checkpoint_continues_eval(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["use_real_api"] = False
    config["experiment"]["disable_llm_during_execution"] = True
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {"enabled": False},
        },
        "allowed_files": ["rosetta/model/aligner.py", "rosetta/model/projector.py", "rosetta/model/wrapper.py"],
        "allowed_prefixes": ["recipe/", "local/auto_research_runs/"],
    }
    paths = init_workspace(config, "topic", project_id="proj_recovery", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        run_repo = Path(working_dir)
        if name == "train":
            run_id = "idea"
            final = run_repo / "local" / "auto_research_runs" / run_id / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
            return {"step": name, "status": "failed", "attempts": [{"stdout": "", "stderr": "train crashed", "returncode": 1}], "returncode": 1}
        if name.startswith("eval_"):
            dataset = name.replace("eval_", "")
            out = run_repo / "local" / "auto_research_runs" / "idea" / "results" / dataset
            out.mkdir(parents=True, exist_ok=True)
            (out / f"Rosetta_{dataset}_generate_summary.json").write_text(
                json.dumps({"model": "Rosetta", "dataset": dataset, "answer_method": "generate", "overall_accuracy": 0.51}),
                encoding="utf-8",
            )
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}

    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    candidate = {
        "id": "idea",
        "title": "Idea",
        "hypothesis": "h",
        "experiment_contract": {"config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}}},
    }
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate=candidate,
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
        proxy_gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )
    assert result["command_status"] in {"partial", "ok"}
    assert result["metrics"]["mean"] == 51.0
    state = json.loads(Path(result["run_state_path"]).read_text(encoding="utf-8"))
    assert any(action["action"] == "skip_failed_train_with_existing_final_checkpoint" for action in state["recovery_actions"])


def test_c2c_train_oom_uses_memory_safe_recipe_then_eval(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "recipe/train_recipe/C2C_0.6+0.5.json").write_text(
        json.dumps(
            {
                "output": {},
                "data": {"kwargs": {}},
                "training": {"learning_rate": 1e-4, "per_device_train_batch_size": 4, "gradient_accumulation_steps": 8},
                "model": {},
            }
        ),
        encoding="utf-8",
    )
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["use_real_api"] = False
    config["experiment"]["disable_llm_during_execution"] = True
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0, 1],
            "full_train_oom_recovery": {
                "enabled": True,
                "per_device_train_batch_size": 1,
                "preserve_effective_batch": True,
            },
            "proxy_screen": {"enabled": False},
        },
        "allowed_files": ["rosetta/model/aligner.py", "rosetta/model/projector.py", "rosetta/model/wrapper.py"],
        "allowed_prefixes": ["recipe/", "local/auto_research_runs/"],
    }
    paths = init_workspace(config, "topic", project_id="proj_oom_memory_safe", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    seen_steps: list[str] = []

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        del retry_policy
        run_repo = Path(working_dir)
        seen_steps.append(name)
        if name in {"train", "train_recovery_reduced_concurrency"}:
            return {
                "step": name,
                "status": "failed",
                "attempts": [{"stdout": "", "stderr": "torch.OutOfMemoryError: CUDA out of memory", "returncode": 1}],
                "returncode": 1,
            }
        if name == "train_recovery_memory_safe":
            assert "train_recipe_memory_safe.json" in command
            run_id = "idea"
            final = run_repo / "local" / "auto_research_runs" / run_id / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
            return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}
        if name == "eval_mmlu-redux":
            out = run_repo / "local" / "auto_research_runs" / "idea" / "results" / "mmlu-redux"
            out.mkdir(parents=True, exist_ok=True)
            (out / "Rosetta_mmlu-redux_generate_summary.json").write_text(
                json.dumps({"model": "Rosetta", "dataset": "mmlu-redux", "answer_method": "generate", "overall_accuracy": 0.51}),
                encoding="utf-8",
            )
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}

    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={
            "id": "idea",
            "title": "Idea",
            "hypothesis": "h",
            "experiment_contract": {"config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}}},
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0, 1], "max_gpus": 2}),
        proxy_gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    memory_safe_config = Path(result["execution_repo"]["repo_root"]) / "local" / "auto_research_runs" / "idea" / "train_recipe_memory_safe.json"
    memory_safe_payload = json.loads(memory_safe_config.read_text(encoding="utf-8"))
    state = json.loads(Path(result["run_state_path"]).read_text(encoding="utf-8"))

    assert seen_steps.index("train_recovery_memory_safe") < seen_steps.index("eval_mmlu-redux")
    assert result["command_status"] == "ok"
    assert result["metrics"]["mean"] == 51.0
    assert memory_safe_payload["training"]["per_device_train_batch_size"] == 1
    assert memory_safe_payload["training"]["gradient_accumulation_steps"] == 16
    assert memory_safe_payload["training"]["learning_rate"] == pytest.approx(1e-4)
    assert any(action["action"] == "retry_train_memory_safe_recipe" and action["recovery_status"] == "ok" for action in state["recovery_actions"])
    memory_safe_action = next(action for action in state["recovery_actions"] if action["action"] == "retry_train_memory_safe_recipe")
    assert memory_safe_action["config_changes"]["learning_rate_adjustment"]["status"] == "unchanged"


def test_deterministic_s3_blocks_noop_candidate(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["experiment"]["disable_llm_during_execution"] = True
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {"enabled": False},
        },
        "allowed_files": ["rosetta/model/aligner.py"],
        "allowed_prefixes": ["recipe/", "local/auto_research_runs/"],
    }
    paths = init_workspace(config, "topic", project_id="proj_noop", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    def fail_run_step(**kwargs):
        raise AssertionError("noop candidate must be blocked before running commands")

    monkeypatch.setattr(agent.runner, "run_step", fail_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={"id": "noop", "title": "Noop"},
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
        proxy_gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    assert result["decision"] == "blocked"
    assert result["command_status"] == "blocked"
    assert result["has_executable_change"] is False


def test_s3_applies_frozen_patch_archives_snapshot_and_does_not_call_llm(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=True)
    config["experiment"]["disable_llm_during_execution"] = True
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {"eval_datasets": ["mmlu-redux"], "train_samples": 1, "gpu_ids": [0]},
    }
    paths = init_workspace(config, "topic", project_id="proj_s3_patch", simulate=True)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))

    class BombLLM:
        use_real_api = True

        def generate(self, **kwargs):
            raise AssertionError("S3 execution must not call LLM")

        def generate_json(self, **kwargs):
            raise AssertionError("S3 execution must not call LLM")

        def generate_json_with_schema(self, **kwargs):
            raise AssertionError("S3 execution must not call LLM")

    context.llm = BombLLM()
    patch_dir = paths.root / "plan/code_patches/frozen_idea"
    patch_dir.mkdir(parents=True)
    aligner = repo / "rosetta/model/aligner.py"
    patch_payload = {
        "schema_version": 1,
        "candidate_id": "frozen_idea",
        "title": "Frozen Idea",
        "operations": [
            {
                "op": "replace_file",
                "path": "rosetta/model/aligner.py",
                "old_sha256": sha256_file(aligner),
                "new": "VALUE = 'frozen patch'\n",
            }
        ],
        "changed_files": ["rosetta/model/aligner.py"],
        "rationale": "Frozen test patch.",
    }
    (patch_dir / "patch.json").write_text(json.dumps(patch_payload), encoding="utf-8")
    candidate = {
        "id": "frozen_idea",
        "title": "Frozen Idea",
        "hypothesis": "h",
        "code_patch": {
            "status": "ok",
            "patch_json": "plan/code_patches/frozen_idea/patch.json",
            "changed_files": ["rosetta/model/aligner.py"],
            "has_executable_change": True,
        },
    }

    agent = ExperimentAgent(context)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate=candidate,
        index=0,
        simulate=True,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
        proxy_gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    snapshot = paths.root / "experiment/code_snapshots/frozen_idea/rosetta/model/aligner.py"
    manifest = paths.root / "experiment/code_snapshots/frozen_idea/manifest.json"
    state = json.loads(Path(result["run_state_path"]).read_text(encoding="utf-8"))

    assert result["patch_result"]["status"] == "applied"
    assert result["code_snapshot"]["status"] == "ok"
    assert result["command_status"] == "mocked"
    assert snapshot.exists()
    assert "frozen patch" in snapshot.read_text(encoding="utf-8")
    assert manifest.exists()
    assert state["code_snapshot"]["status"] == "ok"


def test_s3_prefers_patched_repo_snapshot_over_patch_json(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    original_text = (repo / "rosetta/model/aligner.py").read_text(encoding="utf-8")
    config = _base_config(tmp_path / "workspace", simulate=True)
    config["experiment"]["disable_llm_during_execution"] = True
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {"eval_datasets": ["mmlu-redux"], "train_samples": 1, "gpu_ids": [0]},
    }
    paths = init_workspace(config, "topic", project_id="proj_s3_snapshot", simulate=True)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))

    patch_dir = paths.root / "plan/code_patches/snapshot_idea"
    patched_snapshot = patch_dir / "patched_repo_snapshot"
    shutil.copytree(repo, patched_snapshot)
    (patched_snapshot / "rosetta/model/aligner.py").write_text("VALUE = 'snapshot truth'\n", encoding="utf-8")
    write_json(
        patch_dir / "patched_repo_snapshot_manifest.json",
        {"schema_version": "patched_repo_snapshot_v1", "sha256": "snapshot-sha", "file_count": 1},
    )
    patch_payload = {
        "schema_version": 1,
        "candidate_id": "snapshot_idea",
        "operations": [
            {
                "op": "replace_file",
                "path": "rosetta/model/aligner.py",
                "old_sha256": sha256_file(repo / "rosetta/model/aligner.py"),
                "new": "VALUE = 'patch fallback should not run'\n",
            }
        ],
        "changed_files": ["rosetta/model/aligner.py"],
        "patched_repo_snapshot": {
            "status": "ok",
            "path": "plan/code_patches/snapshot_idea/patched_repo_snapshot",
            "manifest": "plan/code_patches/snapshot_idea/patched_repo_snapshot_manifest.json",
            "sha256": "snapshot-sha",
            "changed_files": ["rosetta/model/aligner.py"],
        },
        "rationale": "Snapshot should be S3 execution truth.",
    }
    write_json(patch_dir / "patch.json", patch_payload)
    candidate = {
        "id": "snapshot_idea",
        "title": "Snapshot Idea",
        "hypothesis": "h",
        "code_patch": {
            "status": "ok",
            "patch_json": "plan/code_patches/snapshot_idea/patch.json",
            "patched_repo_snapshot": patch_payload["patched_repo_snapshot"],
            "changed_files": ["rosetta/model/aligner.py"],
            "has_executable_change": True,
        },
    }

    agent = ExperimentAgent(context)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate=candidate,
        index=0,
        simulate=True,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
        proxy_gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    execution_repo = Path(result["execution_repo"]["repo_root"])
    archived = paths.root / "experiment/code_snapshots/snapshot_idea/rosetta/model/aligner.py"

    assert result["patch_result"]["status"] == "snapshot_applied"
    assert result["execution_repo"]["source"] == "patched_repo_snapshot"
    assert "snapshot truth" in (execution_repo / "rosetta/model/aligner.py").read_text(encoding="utf-8")
    assert "snapshot truth" in archived.read_text(encoding="utf-8")
    assert "patch fallback should not run" not in archived.read_text(encoding="utf-8")
    assert (repo / "rosetta/model/aligner.py").read_text(encoding="utf-8") == original_text


def test_s3_blocks_outputs_written_to_original_snapshot(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["experiment"]["disable_llm_during_execution"] = True
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {"enabled": False},
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_s3_pollution", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        del command, working_dir, retry_policy
        if name == "train":
            final = repo / "local" / "auto_research_runs" / "polluter" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("polluted", encoding="utf-8")
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}

    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={
            "id": "polluter",
            "title": "Polluter",
            "experiment_contract": {"config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}}},
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
        proxy_gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    audit = result["execution_repo_audit"]
    state = json.loads(Path(result["run_state_path"]).read_text(encoding="utf-8"))

    assert result["command_status"] == "blocked"
    assert result["decision"] == "blocked"
    assert result["failure_attribution"]["primary_failure"] == "execution_repo_output_pollution"
    assert audit["status"] == "failed"
    assert "local/auto_research_runs/polluter/checkpoints/final/marker.txt" in audit["output_pollution"]["added_files"]
    assert any(action["action"] == "block_original_snapshot_output_pollution" for action in state["recovery_actions"])


def test_c2c_small_loop_locks_to_selected_patch_manifest_candidate(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["experiment"]["disable_llm_during_execution"] = True
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {"enabled": False},
            "max_candidates": 3,
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_s3_manifest_lock", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    plan = {
        "selected_idea": {"title": "Selected"},
        "candidate_ideas": [
            {
                "id": "local_monotonic_neighbor_window_router",
                "title": "Old Local Monotonic",
                "code_patch": {"status": "ok", "patch_json": "plan/code_patches/local_monotonic_neighbor_window_router/patch.json", "changed_files": ["rosetta/model/aligner.py"], "has_executable_change": True},
                "hypothesis": "old",
            },
            {
                "id": "pathology_triggered_prior_fallback",
                "title": "Selected Pathology",
                "code_patch": {"status": "ok", "patch_json": "plan/code_patches/pathology_triggered_prior_fallback/patch.json", "changed_files": ["rosetta/model/projector.py"], "has_executable_change": True},
                "hypothesis": "new",
            },
        ],
    }
    (paths.root / "plan" / "code_patches").mkdir(parents=True, exist_ok=True)
    selected_patch_dir = paths.root / "plan" / "code_patches" / "pathology_triggered_prior_fallback"
    selected_patch_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        selected_patch_dir / "implementation_contract.json",
        {
            "candidate_id": "pathology_triggered_prior_fallback",
            "hypothesis": "new",
            "mechanism_contract": {"mechanism_type": "pathology_triggered_prior_fallback"},
            "experiment_contract": {"config_overrides": {}, "ablation_switch": "disable_pathology"},
        },
    )
    write_json(
        selected_patch_dir / "patch.json",
        {
            "schema_version": 1,
            "candidate_id": "pathology_triggered_prior_fallback",
            "operations": [],
            "changed_files": ["rosetta/model/projector.py"],
            "implementation_contract": {"candidate_id": "pathology_triggered_prior_fallback"},
        },
    )
    write_json(
        paths.root / "plan" / "code_patches" / "patch_manifest.json",
        {
            "status": "ok",
            "selected_candidate_id": "pathology_triggered_prior_fallback",
            "selected_patch": {
                "candidate_id": "pathology_triggered_prior_fallback",
                "title": "Selected Pathology",
                "status": "ok",
                "patch_json": "plan/code_patches/pathology_triggered_prior_fallback/patch.json",
                "implementation_contract": "plan/code_patches/pathology_triggered_prior_fallback/implementation_contract.json",
                "changed_files": ["rosetta/model/projector.py"],
                "has_executable_change": True,
                "quality_score": {"score": 99},
            },
            "candidates": [
                {
                    "candidate_id": "local_monotonic_neighbor_window_router",
                    "title": "Old Local Monotonic",
                    "status": "ok",
                    "patch_json": "plan/code_patches/local_monotonic_neighbor_window_router/patch.json",
                    "changed_files": ["rosetta/model/aligner.py"],
                    "has_executable_change": True,
                    "quality_score": {"score": 10},
                },
                {
                    "candidate_id": "pathology_triggered_prior_fallback",
                    "title": "Selected Pathology",
                    "status": "ok",
                    "patch_json": "plan/code_patches/pathology_triggered_prior_fallback/patch.json",
                    "implementation_contract": "plan/code_patches/pathology_triggered_prior_fallback/implementation_contract.json",
                    "changed_files": ["rosetta/model/projector.py"],
                    "has_executable_change": True,
                    "quality_score": {"score": 99},
                },
            ],
        },
    )

    seen_candidates: list[str] = []

    def fake_run_single_c2c_candidate(*, candidate, **kwargs):
        seen_candidates.append(candidate["id"])
        return {
            "id": candidate["id"],
            "title": candidate["title"],
            "decision": "candidate_win" if candidate["id"] == "pathology_triggered_prior_fallback" else "not_viable",
            "command_status": "ok",
            "metrics": {"mean": 51.0 if candidate["id"] == "pathology_triggered_prior_fallback" else 49.0},
            "proxy_screen": {"metrics": {"mean": 49.0}, "status": "passed"},
            "activation_smoke": {"status": "passed"},
            "full_s3_readiness": {"status": "ready", "full_train_allowed": False, "worth_full_train": {"decision": "no", "reason": "blocked"}},
            "ablation": {"comparison": {"mechanism_supported": True}},
            "delta_vs_baseline": 1.0 if candidate["id"] == "pathology_triggered_prior_fallback" else -1.0,
            "worst_dataset_regression": 0.0,
            "patch_result": {"status": "ok"},
            "command_logs": [],
            "run_state_path": str(paths.root / "dummy_run_state.json"),
            "has_executable_change": True,
            "decision_reason": "",
        }

    monkeypatch.setattr(agent, "_run_single_c2c_candidate", fake_run_single_c2c_candidate)
    result = agent._run_c2c_small_loop(
        plan,
        {"baseline": {"mean": 50.0}, "max_candidates": 3, "min_delta_to_pass": 0.1, "max_dataset_regression": 2.0, "selected_gpu_ids": [0], "gpu_policy": {}},
        "env.md",
        None,
    )

    selection = json.loads((paths.root / "experiment" / "results" / "s3_candidate_selection.json").read_text(encoding="utf-8"))
    assert seen_candidates == ["pathology_triggered_prior_fallback"]
    assert selection["mode"] == "patch_manifest_selected"
    assert selection["executed_candidate_ids"] == ["pathology_triggered_prior_fallback"]
    assert selection["skipped_candidate_ids"] == ["local_monotonic_neighbor_window_router"]
    assert selection["patch_manifest"]["sha256"]
    assert selection["selected_patch"]["sha256"] == sha256_file(selected_patch_dir / "patch.json")
    assert selection["selected_implementation_contract"]["sha256"] == sha256_file(selected_patch_dir / "implementation_contract.json")
    assert result["artifacts"]


def test_s3_runs_ablation_switch_disabled_eval(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["experiment"]["disable_llm_during_execution"] = True
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "replay",
                "eval_datasets": ["mmlu-redux"],
                "eval_limit": 1,
                "train_samples": 1,
                "require_proxy_metrics": True,
                "require_paired_baseline": True,
                "run_baseline_if_missing": True,
                "min_proxy_mean_delta": -0.3,
                "activation_smoke": {"enabled": True, "max_datasets": 1, "min_abs_metric_delta": 0.01},
            },
        },
        "allowed_prefixes": ["local/auto_research_runs/"],
    }
    paths = init_workspace(config, "topic", project_id="proj_s3_ablation", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    seen_steps: list[str] = []

    def write_summary(root: Path, dataset: str, accuracy: float) -> None:
        out = root / dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / f"Rosetta_{dataset}_generate_summary.json").write_text(
            json.dumps({"model": "Rosetta", "dataset": dataset, "answer_method": "generate", "overall_accuracy": accuracy}),
            encoding="utf-8",
        )

    def write_predictions(root: Path, dataset: str, labels: list[str]) -> None:
        out = root / dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / "prediction_outputs.jsonl").write_text(
            "\n".join(json.dumps({"prediction": f"Answer: {label}", "answer": label}) for label in labels) + "\n",
            encoding="utf-8",
        )

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        del command, retry_policy
        seen_steps.append(name)
        run_root = Path(working_dir) / "local" / "auto_research_runs" / "mechanism"
        if name == "proxy_baseline_train":
            final = Path(working_dir) / "local" / "auto_research_runs" / "proxy_baseline" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        elif name == "proxy_baseline_eval_mmlu-redux":
            write_summary(Path(working_dir) / "local" / "auto_research_runs" / "proxy_baseline" / "results", "mmlu-redux", 0.50)
        elif name == "proxy_command_0":
            final = run_root / "proxy" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        elif name == "proxy_command_1":
            root = run_root / "proxy" / "results"
            write_summary(root, "mmlu-redux", 0.505)
            write_predictions(root, "mmlu-redux", ["A", "B", "C", "D"])
        elif name == "activation_smoke_eval_mmlu-redux":
            root = run_root / "proxy" / "activation_smoke_disabled" / "results"
            write_summary(root, "mmlu-redux", 0.49)
            write_predictions(root, "mmlu-redux", ["B", "B", "C", "D"])
        elif name == "train":
            final = run_root / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        elif name.startswith("eval_"):
            dataset = name.replace("eval_", "")
            write_summary(run_root / "results", dataset, 0.55)
        elif name.startswith("ablation_eval_"):
            dataset = name.replace("ablation_eval_", "")
            write_summary(run_root / "ablation_disabled" / "results", dataset, 0.50)
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}

    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    candidate = {
        "id": "mechanism",
        "title": "Mechanism",
        "hypothesis": "h",
        "experiment_contract": {
            "ablation_switch": "disable_mechanism",
            "config_overrides": {
                "train": {"model": {"mechanism_enabled": True}},
                "eval": {"model": {"rosetta_config": {"mechanism_enabled": True}}},
            },
        },
    }

    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate=candidate,
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
        proxy_gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    ablation = result["ablation"]
    comparison = ablation["comparison"]
    disabled_eval = yaml.safe_load((Path(result["execution_repo"]["repo_root"]) / "local/auto_research_runs/mechanism/ablation_disabled/eval_mmlu-redux.yaml").read_text(encoding="utf-8"))

    assert "ablation_eval_mmlu-redux" in seen_steps
    assert result["metrics"]["mean"] == 55.0
    assert ablation["status"] == "ok"
    assert ablation["metrics"]["mean"] == 50.0
    assert comparison["enabled_minus_disabled_mean"] == 5.0
    assert comparison["mechanism_supported"] is True
    assert disabled_eval["model"]["rosetta_config"]["disable_mechanism"] is True
    assert disabled_eval["output"]["output_dir"] == "local/auto_research_runs/mechanism/ablation_disabled/results/mmlu-redux"


def test_c2c_ablation_payload_and_verification_report_supported(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=True)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
    }
    adapter = C2CAdapter(tmp_path / "project", config)
    best = {
        "id": "mechanism",
        "title": "Mechanism",
        "decision": "candidate_win",
        "command_status": "ok",
        "metrics": {"mean": 55.0, "datasets": {"mmlu-redux": 55.0}},
        "delta_vs_baseline": 5.0,
        "worst_dataset_regression": 0.0,
        "ablation": {
            "enabled": True,
            "status": "ok",
            "switch": "disable_mechanism",
            "metrics": {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
            "comparison": {
                "status": "ok",
                "enabled_mean": 55.0,
                "disabled_mean": 50.0,
                "enabled_minus_disabled_mean": 5.0,
                "dataset_enabled_minus_disabled": {"mmlu-redux": 5.0},
                "mechanism_supported": True,
            },
        },
    }
    payload = {
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "best_candidate": best,
        "candidate_results": [best],
    }

    ablation_payload = ExperimentAgent._c2c_ablation_payload(payload, adapter)
    verification = ExperimentAgent._c2c_verification_md(best, 50.0, [best], 0.1, 2.0)

    assert ablation_payload["status"] == "ok"
    assert ablation_payload["best_supported"] is True
    assert ablation_payload["best_delta_enabled_vs_disabled"] == 5.0
    assert ablation_payload["candidate_ablations"][0]["supported"] is True
    assert "H2: supported" in verification
    assert "disable_mechanism" in verification


def test_c2c_ablation_payload_distinguishes_declared_switch_from_reached_stage(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=True)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
    }
    adapter = C2CAdapter(tmp_path / "project", config)
    candidate = {
        "id": "proxy_rejected",
        "title": "Proxy Rejected",
        "decision": "proxy_rejected",
        "command_status": "proxy_rejected",
        "experiment_contract": {"ablation_switch": "disable_proxy_rejected"},
        "metrics": None,
        "ablation": {"enabled": False, "status": "skipped", "reason": "not run"},
    }

    ablation_payload = ExperimentAgent._c2c_ablation_payload(
        {"baseline": config["c2c"]["baseline"], "best_candidate": None, "candidate_results": [candidate]},
        adapter,
    )

    assert ablation_payload["status"] == "skipped"
    assert ablation_payload["reason"] == "candidate ablation switches were declared, but no candidate reached full eval before ablation"
    assert ablation_payload["candidate_ablations"][0]["declared_switch"] == "disable_proxy_rejected"
    assert ablation_payload["candidate_ablations"][0]["reached_ablation_stage"] is False


def test_c2c_failure_analysis_accepts_grouped_posthoc_suggestions() -> None:
    payload = {
        "acceptance": {"passed": False, "reason": "no candidate metrics"},
        "best_candidate": None,
    }
    posthoc = {
        "failure_modes": [{"observed": "Patch rejected before preflight."}],
        "next_round_suggestions": {
            "S1": [{"constraint": "Only submit contract-safe candidates."}],
            "S2.5": [{"action": "Keep new knobs fixed or explicitly activated."}],
        },
        "avoid_repeat_rules": {"S2": [{"rule": "Do not repeat config-unsafe patches."}]},
    }

    markdown = ExperimentAgent._c2c_failure_analysis_md(payload, posthoc)

    assert "Patch rejected before preflight." in markdown
    assert "S1: Only submit contract-safe candidates." in markdown
    assert "S2.5: Keep new knobs fixed or explicitly activated." in markdown
    assert "S2: Do not repeat config-unsafe patches." in markdown


def test_s3_rejects_validation_failed_code_patch_before_training(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {"eval_datasets": ["mmlu-redux"], "train_samples": 1, "gpu_ids": [0]},
    }
    paths = init_workspace(config, "topic", project_id="proj_s3_bad_patch", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    def fail_run_step(**kwargs):
        raise AssertionError("validation_failed patch must be blocked before training")

    monkeypatch.setattr(agent.runner, "run_step", fail_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={
            "id": "bad_patch",
            "title": "Bad Patch",
            "code_patch": {"status": "validation_failed", "reason": "py_compile failed"},
            "experiment_contract": {"config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}}},
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
        proxy_gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    assert result["decision"] == "patch_rejected"
    assert result["command_status"] == "patch_rejected"
    assert result["patch_result"]["patch_status"] == "validation_failed"
    assert "py_compile failed" in result["patch_result"]["errors"][0]


def test_c2c_static_proxy_rejects_evaluator_patch_before_training(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "static",
                "reject_eval_code_changes": True,
                "reject_if_no_executable_change": True,
            },
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_proxy_static", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    patch_dir = paths.root / "plan/code_patches/eval_risk"
    patch_dir.mkdir(parents=True)
    evaluator = repo / "script/evaluation/unified_evaluator.py"
    (patch_dir / "patch.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_id": "eval_risk",
                "title": "Eval Risk",
                "operations": [
                    {
                        "op": "replace_file",
                        "path": "script/evaluation/unified_evaluator.py",
                        "old_sha256": sha256_file(evaluator),
                        "new": "print('changed eval')\n",
                    }
                ],
                "changed_files": ["script/evaluation/unified_evaluator.py"],
            }
        ),
        encoding="utf-8",
    )
    agent = ExperimentAgent(context)
    seen_steps = []

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        seen_steps.append(name)
        if name == "train" or name.startswith("eval_"):
            raise AssertionError("proxy rejected candidate must not reach full train/eval")
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}

    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={
            "id": "eval_risk",
            "title": "Eval Risk",
            "code_patch": {
                "status": "ok",
                "patch_json": "plan/code_patches/eval_risk/patch.json",
                "changed_files": ["script/evaluation/unified_evaluator.py"],
                "has_executable_change": True,
            },
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
        proxy_gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    state = json.loads(Path(result["run_state_path"]).read_text(encoding="utf-8"))
    assert result["decision"] == "proxy_repairable"
    assert result["command_status"] == "proxy_repairable"
    assert result["proxy_screen"]["status"] == "repairable_proxy_risk"
    assert result["failure_attribution"]["primary_failure"] == "repairable_proxy_risk_before_full_training"
    assert "evaluation_code_changed" in result["failure_attribution"]["patch_risk"]["risk_labels"]
    assert "train" not in seen_steps
    assert state["train"] is None


def test_c2c_proxy_carries_instrumentation_quality_repair_request(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "static",
                "require_proxy_metrics": False,
                "require_paired_baseline": False,
                "soft_proxy_mean_delta": None,
                "soft_max_proxy_dataset_regression": None,
                "soft_min_proxy_score": None,
                "activation_smoke": {"enabled": False},
            },
        },
    }
    config["code_patch"] = {
        "enabled": True,
        "dynamic_whitelist": {
            "include_prefixes": ["rosetta/"],
            "include_extensions": [".py"],
            "exclude_prefixes": [],
            "exclude_extensions": [],
            "include_root_globs": [],
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_quality_repair_proxy", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    patch_dir = paths.root / "plan" / "code_patches" / "quality"
    patch_dir.mkdir(parents=True)
    projector = repo / "rosetta/model/projector.py"
    (patch_dir / "patch.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_id": "quality",
                "title": "Quality",
                "operations": [
                    {
                        "op": "replace_file",
                        "path": "rosetta/model/projector.py",
                        "old_sha256": sha256_file(projector),
                        "new": "VALUE = 'quality runnable mechanism'\n",
                    }
                ],
                "changed_files": ["rosetta/model/projector.py"],
                "mechanism_review": {
                    "status": "ok",
                    "soft_issues": ["missing_coverage_diagnostics_evidence"],
                    "quality_repair": {
                        "needed": False,
                        "deferred": True,
                        "mode": "paperization_after_effect",
                        "issues": ["missing_coverage_diagnostics_evidence"],
                        "constraints": ["Only add instrumentation."],
                        "ablation_switch": "ablation_disable_quality",
                    },
                },
                "quality_score": {"score": 70, "soft_issues": ["missing_coverage_diagnostics_evidence"]},
            }
        ),
        encoding="utf-8",
    )
    agent = ExperimentAgent(context)
    monkeypatch.setattr(agent.runner, "run_step", lambda **kwargs: {"step": kwargs["name"], "status": "ok", "returncode": 0, "stdout": "", "stderr": "", "attempts": []})
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={
            "id": "quality",
            "title": "Quality",
            "code_patch": {
                "status": "ok",
                "patch_json": "plan/code_patches/quality/patch.json",
                "changed_files": ["rosetta/model/projector.py"],
                "has_executable_change": True,
            },
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
        proxy_gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    assert result["proxy_screen"]["status"] == "repairable_proxy_risk"
    assert result["proxy_screen"]["repair_mode"] == "full_s3_readiness_repair"
    assert result["proxy_screen"]["quality_repair"]["needed"] is False
    assert result["proxy_screen"]["quality_repair"]["deferred"] is True
    assert result["proxy_screen"]["quality_repair"]["repair_route"] == "paperization"
    assert result["proxy_screen"]["quality_repair"]["mode"] == "paperization_after_effect"
    assert result["failure_attribution"]["quality_repair"]["acceptance_guard"]["rerun_same_proxy_subset"] is True


@pytest.mark.parametrize("bootstrap", [False, True], ids=["standard", "bootstrap"])
def test_s3_reuses_completed_proxy_rejected_run_state_without_rerun(monkeypatch, tmp_path: Path, bootstrap: bool) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "replay",
                "train_samples": 1,
                "eval_limit": 1,
                "eval_datasets": ["mmlu-redux"],
                "min_proxy_mean_delta": -0.3,
                "require_paired_baseline": True,
                "baseline_cache_path": "experiment/results/c2c_proxy_baseline.json",
            },
        },
        "allowed_files": ["rosetta/model/projector.py"],
        "allowed_prefixes": ["recipe/", "local/auto_research_runs/"],
    }
    if bootstrap:
        config["orchestration"] = {"profile": "bootstrap", "bootstrap": {"proxy_only": True}}
    paths = init_workspace(config, "topic", project_id="proj_proxy_resume_reuse", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    adapter = C2CAdapter(paths.root, config)
    gpu_selection = agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1})
    candidate = {
        "id": "proxy_resume",
        "title": "Proxy Resume",
        "experiment_contract": {"config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}}},
    }
    patch = {"operations": [], "changed_files": [], "summary": "config-only cached proxy test"}
    execution_repo = agent._prepare_c2c_execution_repo(candidate, adapter, patch)
    execution_adapter = C2CAdapter(paths.root, {**config, "c2c": {**config["c2c"], "snapshot_path": execution_repo["repo_root"]}})
    run_spec = execution_adapter.materialize_candidate_configs(candidate, gpu_selection)
    patch_fingerprint = ExperimentAgent._c2c_patch_fingerprint(
        execution_adapter,
        {"status": "skipped", "changed_files": [], "execution_repo": execution_repo},
        run_spec,
    )
    state_path = Path(run_spec["run_state_path"])
    state_path.write_text(
        json.dumps(
            {
                "candidate_id": "proxy_resume",
                "run_id": "proxy_resume",
                "preflight": {"status": "ok"},
                "proxy_screen": {
                    "enabled": True,
                    "status": "rejected",
                    "reason": "proxy mean delta -1.0 below hard threshold -0.3",
                    "metrics": {"mean": 49.0, "datasets": {"mmlu-redux": 49.0}},
                    "baseline_metrics": {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
                    "proxy_delta_vs_baseline": -1.0,
                    "proxy_dataset_deltas": {"mmlu-redux": -1.0},
                    "proxy_dataset_regressions": {"mmlu-redux": 1.0},
                    "proxy_worst_dataset_regression": 1.0,
                    "proxy_score": -1.5,
                    "patch_fingerprint": patch_fingerprint,
                },
                "metrics": None,
                "attempts": [],
                "frozen_hashes": run_spec["frozen_hashes"],
                "config_overrides": run_spec["config_overrides"],
                "has_executable_change": True,
                "patch_fingerprint": patch_fingerprint,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def fail_run_step(**kwargs):
        raise AssertionError("completed proxy_rejected run_state should be reused")

    monkeypatch.setattr(agent.runner, "run_step", fail_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=adapter,
        candidate=candidate,
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=gpu_selection,
        proxy_gpu_selection=gpu_selection,
    )

    assert result["decision"] == ("bootstrap_proxy_complete" if bootstrap else "proxy_rejected")
    assert result["command_status"] == ("bootstrap_proxy_complete" if bootstrap else "proxy_rejected")
    assert result["proxy_screen"]["proxy_delta_vs_baseline"] == -1.0
    assert result["proxy_screen"]["comparison_baseline_mean"] == 50.0
    assert result["proxy_screen"]["proxy_delta_vs_comparison_baseline"] == -1.0
    assert result["proxy_screen"]["artifact_paths"]["run_state"].endswith("run_state.json")
    assert result["patch_fingerprint"] == patch_fingerprint
    assert result["metrics"] is None
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["proxy_screen"]["status"] == "rejected"
    assert saved["patch_fingerprint"] == patch_fingerprint
    if bootstrap:
        assert saved["bootstrap"]["status"] == "proxy_reached"
        assert saved["bootstrap"]["original_proxy_status"] == "rejected"


def test_s3_proxy_reuse_requires_matching_patch_fingerprint(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {"eval_datasets": ["mmlu-redux"], "train_samples": 1, "gpu_ids": [0]},
        "allowed_files": ["rosetta/model/projector.py"],
        "allowed_prefixes": ["recipe/", "local/auto_research_runs/"],
    }
    paths = init_workspace(config, "topic", project_id="proj_proxy_resume_fingerprint", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    adapter = C2CAdapter(paths.root, config)
    gpu_selection = agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1})
    candidate = {
        "id": "proxy_resume",
        "title": "Proxy Resume",
        "experiment_contract": {"config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}}},
    }
    run_spec = adapter.materialize_candidate_configs(candidate, gpu_selection)
    state_path = Path(run_spec["run_state_path"])
    state_path.write_text(
        json.dumps(
            {
                "candidate_id": "proxy_resume",
                "run_id": "proxy_resume",
                "preflight": {"status": "ok"},
                "proxy_screen": {
                    "enabled": True,
                    "status": "rejected",
                    "metrics": {"mean": 49.0, "datasets": {"mmlu-redux": 49.0}},
                    "patch_fingerprint": "old_patch",
                },
                "metrics": None,
                "frozen_hashes": run_spec["frozen_hashes"],
                "patch_fingerprint": "old_patch",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert agent._load_reusable_c2c_proxy_state(run_spec, "new_patch") is None


def test_c2c_proxy_metric_near_threshold_is_repairable() -> None:
    decision = ExperimentAgent._c2c_proxy_metric_decision(
        metrics={"mean": 49.8, "datasets": {"mmlu-redux": 49.8}},
        baseline={"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        proxy_baseline={"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        patch_risk={"risk_labels": []},
        proxy_cfg={
            "require_paired_baseline": True,
            "min_proxy_mean_delta": -0.1,
            "repairable_proxy_mean_margin": 0.25,
            "max_proxy_dataset_regression": 1.5,
        },
    )

    assert decision["status"] == "repairable_proxy_risk"
    assert decision["repair_route"] == "S2_plan"
    assert decision["proxy_delta_vs_baseline"] == -0.2
    assert decision["full_baseline_mean"] == 50.0
    assert decision["proxy_baseline_mean"] == 50.0
    assert decision["comparison_baseline_mean"] == 50.0
    assert decision["proxy_delta_vs_comparison_baseline"] == -0.2
    assert decision["proxy_delta_vs_proxy_baseline"] == -0.2


def test_c2c_proxy_soft_zero_delta_is_repairable_when_configured() -> None:
    decision = ExperimentAgent._c2c_proxy_metric_decision(
        metrics={"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        baseline={"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        proxy_baseline={"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        patch_risk={"risk_labels": []},
        proxy_cfg={
            "require_paired_baseline": True,
            "min_proxy_mean_delta": -0.3,
            "soft_proxy_mean_delta": 0.1,
            "repair_soft_proxy_fail": True,
            "max_proxy_dataset_regression": 1.5,
        },
    )

    assert decision["status"] == "repairable_proxy_risk"
    assert decision["soft_fail"] is True
    assert decision["repair_route"] == "S2_plan"
    assert decision["repair_mode"] == "effect_first_proxy_repair"
    assert "paperization" in decision["repair_hint"]
    assert "proxy mean delta" in decision["reason"]
    repair_contract = decision["proxy_effect_repair_contract"]
    assert repair_contract["mode"] == "effect_first_proxy_repair"
    assert repair_contract["proxy_delta_vs_baseline"] == 0.0
    assert repair_contract["proxy_delta_vs_comparison_baseline"] == 0.0
    assert repair_contract["proxy_baseline_mean"] == 50.0
    assert "proxy mean delta" in repair_contract["soft_flags"][0]
    assert any("paperization" in item for item in repair_contract["forbidden"])


def test_c2c_proxy_soft_warning_passes_by_default() -> None:
    decision = ExperimentAgent._c2c_proxy_metric_decision(
        metrics={"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        baseline={"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        proxy_baseline={"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        patch_risk={"risk_labels": []},
        proxy_cfg={
            "require_paired_baseline": True,
            "min_proxy_mean_delta": -0.3,
            "soft_proxy_mean_delta": 0.1,
            "max_proxy_dataset_regression": 1.5,
        },
    )

    assert decision["status"] == "passed"
    assert decision["soft_fail"] is True
    assert "soft warnings" in decision["reason"]
    assert "proxy mean delta" in decision["soft_flags"][0]


def test_c2c_proxy_zero_delta_with_patch_risk_passes_lowered_effect_first_threshold() -> None:
    decision = ExperimentAgent._c2c_proxy_metric_decision(
        metrics={"mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0}},
        baseline={"mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0}},
        proxy_baseline={"mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0}},
        patch_risk={"risk_labels": ["alignment_mechanism_changed", "projector_mechanism_changed", "training_loop_changed", "config_override_changed", "test_change"]},
        proxy_cfg={
            "require_paired_baseline": True,
            "min_proxy_mean_delta": -0.3,
            "soft_proxy_mean_delta": -0.1,
            "max_proxy_dataset_regression": 1.5,
            "soft_max_proxy_dataset_regression": 0.75,
            "proxy_score_regression_weight": 0.5,
            "risk_penalty_per_label": 0.05,
            "soft_min_proxy_score": -0.3,
        },
    )

    assert decision["status"] == "passed"
    assert decision["proxy_delta_vs_proxy_baseline"] == 0.0
    assert decision["proxy_score"] == -0.25
    assert not decision.get("soft_fail")


def test_c2c_neutral_proxy_policy_allows_small_negative_but_blocks_clear_regression() -> None:
    cfg = {
        "allow_neutral_proxy_full_s3": True,
        "neutral_proxy_min_delta": -0.1,
        "neutral_proxy_max_dataset_regression": 0.25,
    }

    assert experiment_module._c2c_neutral_proxy_full_s3_allowed(
        {"status": "passed", "proxy_delta_vs_comparison_baseline": -0.05, "proxy_worst_dataset_regression": 0.2},
        cfg,
    )
    assert not experiment_module._c2c_neutral_proxy_full_s3_allowed(
        {"status": "passed", "proxy_delta_vs_comparison_baseline": -0.2, "proxy_worst_dataset_regression": 0.0},
        cfg,
    )
    assert not experiment_module._c2c_neutral_proxy_full_s3_allowed(
        {"status": "passed", "proxy_delta_vs_comparison_baseline": 0.0, "proxy_worst_dataset_regression": 0.3},
        cfg,
    )


def test_c2c_proxy_positive_mean_with_borderline_dataset_regression_passes_with_warning() -> None:
    decision = ExperimentAgent._c2c_proxy_metric_decision(
        metrics={
            "mean": 39.2497,
            "datasets": {
                "ai2-arc": 39.9471,
                "mmlu-redux": 37.4375,
                "openbookqa": 40.3646,
            },
        },
        baseline={
            "mean": 50.06,
            "datasets": {
                "ai2-arc": 42.0,
                "mmlu-redux": 42.0,
                "openbookqa": 52.6,
            },
        },
        proxy_baseline={
            "mean": 38.3648,
            "datasets": {
                "ai2-arc": 38.0952,
                "mmlu-redux": 35.8533,
                "openbookqa": 41.1458,
            },
        },
        patch_risk={"risk_labels": []},
        proxy_cfg={
            "require_paired_baseline": True,
            "min_proxy_mean_delta": -0.3,
            "soft_proxy_mean_delta": 0.0,
            "max_proxy_dataset_regression": 1.5,
            "soft_max_proxy_dataset_regression": 0.75,
            "soft_min_proxy_score": 0.0,
            "proxy_score_regression_weight": 0.5,
        },
    )

    assert decision["status"] == "passed"
    assert decision["soft_fail"] is True
    assert decision["proxy_delta_vs_proxy_baseline"] == 0.8849
    assert decision["proxy_worst_dataset_regression"] == 0.7812
    assert "soft warnings" in decision["reason"]
    assert "proxy worst dataset regression" in decision["soft_flags"][0]


def test_c2c_cached_proxy_soft_repairable_rejudges_against_current_config() -> None:
    cached_proxy = {
        "status": "repairable_proxy_risk",
        "reason": "proxy worst dataset regression 0.7812 above soft threshold 0.75",
        "metrics": {
            "mean": 39.2497,
            "datasets": {
                "ai2-arc": 39.9471,
                "mmlu-redux": 37.4375,
                "openbookqa": 40.3646,
            },
        },
        "baseline_metrics": {
            "mean": 38.3648,
            "datasets": {
                "ai2-arc": 38.0952,
                "mmlu-redux": 35.8533,
                "openbookqa": 41.1458,
            },
        },
    }

    decision = ExperimentAgent._c2c_rejudge_cached_proxy_screen(
        cached_proxy,
        baseline={
            "mean": 50.06,
            "datasets": {
                "ai2-arc": 42.0,
                "mmlu-redux": 42.0,
                "openbookqa": 52.6,
            },
        },
        proxy_cfg={
            "require_paired_baseline": True,
            "min_proxy_mean_delta": -0.3,
            "soft_proxy_mean_delta": 0.0,
            "max_proxy_dataset_regression": 1.5,
            "soft_max_proxy_dataset_regression": 0.75,
            "soft_min_proxy_score": 0.0,
            "proxy_score_regression_weight": 0.5,
            "repair_soft_proxy_fail": False,
        },
    )

    assert decision is not None
    assert decision["status"] == "passed"
    assert decision["soft_fail"] is True
    assert decision["proxy_delta_vs_proxy_baseline"] == 0.8849


def test_c2c_proxy_metric_fallback_names_full_and_comparison_baseline() -> None:
    decision = ExperimentAgent._c2c_proxy_metric_decision(
        metrics={"mean": 49.5, "datasets": {"mmlu-redux": 49.5}},
        baseline={"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        proxy_baseline=None,
        patch_risk={"risk_labels": []},
        proxy_cfg={
            "require_paired_baseline": False,
            "min_proxy_mean_delta": -1.0,
            "max_proxy_dataset_regression": 1.5,
        },
    )

    assert decision["status"] == "passed"
    assert decision["proxy_decision_mode"] == "configured_full_baseline"
    assert decision["full_baseline_mean"] == 50.0
    assert decision["proxy_baseline"] is None
    assert decision["proxy_baseline_mean"] is None
    assert decision["comparison_baseline_mean"] == 50.0
    assert decision["proxy_delta_vs_comparison_baseline"] == -0.5
    assert decision["proxy_delta_vs_proxy_baseline"] is None


def test_c2c_proxy_command_failure_classifies_runtime_errors() -> None:
    dtype_failure = ExperimentAgent._c2c_proxy_command_failure(
        {
            "step": "proxy_command_0",
            "returncode": 1,
            "attempts": [
                {
                    "stdout": "",
                    "stderr": "RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16",
                }
            ],
        }
    )
    shape_failure = ExperimentAgent._c2c_proxy_command_failure(
        {
            "step": "proxy_command_0",
            "returncode": 1,
            "attempts": [{"stdout": "", "stderr": "TypeError: must be real number, not list"}],
        }
    )

    assert dtype_failure["category"] == "dtype_mismatch"
    assert "dtype/device" in dtype_failure["repair_hint"]
    assert shape_failure["category"] == "schema_shape_mismatch"


def test_c2c_proxy_command_failure_classifies_oom_as_resource_retry() -> None:
    failure = ExperimentAgent._c2c_proxy_command_failure(
        {
            "step": "proxy_command_0",
            "returncode": 1,
            "attempts": [
                {
                    "stdout": "Starting training",
                    "stderr": "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.00 MiB.",
                    "elapsed_seconds": 373.0,
                    "timeout_seconds": 1800,
                }
            ],
        }
    )

    assert failure["category"] == "resource_oom"
    assert "do not repair the S2.5 patch" in failure["repair_hint"]
    assert failure["elapsed_seconds"] == 373.0


def test_normalize_c2c_proxy_screen_promotes_legacy_oom_to_resource_retry() -> None:
    normalized = experiment_module._normalize_c2c_proxy_screen_artifacts(
        {
            "enabled": True,
            "status": "repairable_proxy_risk",
            "reason": "proxy command 0 failed: CUDA out of memory",
            "command_failure": {
                "category": "resource_oom",
                "summary": "torch.OutOfMemoryError: CUDA out of memory",
            },
        },
        full_baseline={"mean": 50.0, "datasets": {}},
        run_spec={"run_state_path": "run_state.json", "proxy_screen": {"metrics_path": "proxy.json"}},
    )

    assert normalized["status"] == "resource_retry"
    assert normalized["resource_retry"] is True
    assert normalized["failure_category"] == "s3_proxy_resource_oom"
    assert normalized["repair_route"] == "resource_retry"
    assert "do not repair the S2.5 patch" in normalized["repair_hint"]


def test_c2c_proxy_command_failure_classifies_timeout() -> None:
    failure = ExperimentAgent._c2c_proxy_command_failure(
        {
            "step": "proxy_command_1",
            "returncode": 124,
            "attempts": [
                {
                    "stdout": "started",
                    "stderr": "Command timed out after 1200s",
                    "timed_out": True,
                    "elapsed_seconds": 1200.1,
                    "timeout_seconds": 1200,
                }
            ],
        }
    )

    assert failure["category"] == "proxy_timeout"
    assert "inference/training cost" in failure["repair_hint"]
    assert failure["timeout_seconds"] == 1200


def test_c2c_proxy_eval_timeout_default_allows_long_single_card_eval() -> None:
    assert DEFAULT_C2C_PROXY_SCREEN["eval_timeout_seconds"] == 7200
    assert DEFAULT_C2C_PROXY_SCREEN["gpu_policy"]["max_gpus"] == 1


def test_c2c_proxy_command_failure_classifies_distributed_child_failure() -> None:
    failure = ExperimentAgent._c2c_proxy_command_failure(
        {
            "step": "proxy_command_0",
            "returncode": 1,
            "attempts": [
                {
                    "stdout": "training started",
                    "stderr": (
                        "torch.distributed.elastic.multiprocessing.errors.ChildFailedError:\n"
                        "Root Cause (first observed failure):\n"
                        "  rank      : 3 (local_rank: 3)\n"
                        "  exitcode  : 1 (pid: 12345)\n"
                    ),
                    "elapsed_seconds": 42.0,
                }
            ],
        }
    )

    assert failure["category"] == "distributed_child_failed"
    assert "single auto-selected GPU" in failure["repair_hint"]
    assert failure["rank_failure"] == {"local_rank": 3, "exitcode": 1, "pid": 12345}
    assert "ChildFailedError" in failure["stderr_tail"]


def test_c2c_result_payload_compacts_patch_state_and_proxy_logs(tmp_path: Path, monkeypatch) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "replay",
                "eval_datasets": ["mmlu-redux"],
                "train_samples": 1,
                "eval_limit": 1,
                "reject_on_command_failure": True,
            },
        },
    }
    config["code_patch"] = {
        "enabled": True,
        "dynamic_whitelist": {
            "include_prefixes": ["rosetta/", "script/", "recipe/", "test/", "tests/"],
            "include_extensions": [".py", ".json", ".yaml", ".yml", ".toml", ".txt"],
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_compact_payload", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    adapter = C2CAdapter(paths.root, config)

    patch_path = paths.root / "plan" / "code_patches" / "compact" / "patch.json"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    old_sha = sha256_file(adapter.repo_root / "rosetta/model/aligner.py")
    patch_path.write_text(
        json.dumps(
            {
                "operations": [
                    {
                        "op": "replace_file",
                        "path": "rosetta/model/aligner.py",
                        "old_sha256": old_sha,
                        "new": "VALUE = 'compact payload patch'\n",
                    }
                ],
                "changed_files": ["rosetta/model/aligner.py"],
            }
        ),
        encoding="utf-8",
    )

    def fake_preflight(run_spec, gpu_selection):
        del gpu_selection
        payload = {"status": "ok", "checks": [], "errors": [], "warnings": [], "recovery_actions": []}
        Path(run_spec["preflight_path"]).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        del command, retry_policy
        run_repo = Path(working_dir)
        if name.startswith("preflight_command_") or name.startswith("proxy_baseline_"):
            if name == "proxy_baseline_train":
                final = run_repo / "local" / "auto_research_runs" / "proxy_baseline" / "checkpoints" / "final"
                final.mkdir(parents=True, exist_ok=True)
                (final / "marker.txt").write_text("ok", encoding="utf-8")
            if name == "proxy_baseline_eval_mmlu-redux":
                out = run_repo / "local" / "auto_research_runs" / "proxy_baseline" / "results" / "mmlu-redux"
                out.mkdir(parents=True, exist_ok=True)
                (out / "Rosetta_mmlu-redux_generate_summary.json").write_text(
                    json.dumps({"model": "Rosetta", "dataset": "mmlu-redux", "answer_method": "generate", "overall_accuracy": 0.5}),
                    encoding="utf-8",
                )
            return {"step": name, "status": "ok", "returncode": 0, "stdout": "ok", "stderr": "", "attempts": []}
        return {
            "step": name,
            "status": "failed",
            "returncode": 1,
            "stdout": "x" * 9000,
            "stderr": "RuntimeError: compact failure\n" + "y" * 9000,
            "attempts": [{"stdout": "nested" * 1000, "stderr": "nested-err" * 1000}],
        }

    monkeypatch.setattr(adapter, "preflight", fake_preflight)
    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=adapter,
        candidate={
            "id": "compact",
            "title": "Compact",
            "code_patch": {
                "status": "ok",
                "patch_json": "plan/code_patches/compact/patch.json",
                "changed_files": ["rosetta/model/aligner.py"],
                "has_executable_change": True,
            },
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
        proxy_gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    rendered = json.dumps(result, ensure_ascii=False)
    command_log = json.loads((paths.root / "experiment" / "logs" / "c2c_compact_commands.json").read_text(encoding="utf-8"))

    assert "restore_state" not in result["patch_result"]
    assert result["patch_result"]["restore_state_omitted"] is True
    assert len(rendered) < 30000
    assert "x" * 5000 not in rendered
    assert "y" * 5000 not in rendered
    assert "stdout_tail" in result["proxy_screen"]["attempts"][0]
    assert "stdout" not in result["proxy_screen"]["attempts"][0]
    assert "stdout_tail" in command_log["runs"][0]
    assert "stdout" not in command_log["runs"][0]


def test_c2c_replay_proxy_runs_paired_baseline_before_full_training(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "replay",
                "eval_datasets": ["mmlu-redux"],
                "eval_limit": 2,
                "train_samples": 2,
                "require_proxy_metrics": True,
                "require_paired_baseline": True,
                "run_baseline_if_missing": True,
                "min_proxy_mean_delta": -0.3,
                "max_proxy_dataset_regression": 1.5,
                "activation_smoke": {"enabled": True, "max_datasets": 1, "min_abs_metric_delta": 0.01},
            },
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_proxy_replay", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    seen_steps = []

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        run_repo = Path(working_dir)
        seen_steps.append(name)
        if name == "proxy_baseline_train":
            final = run_repo / "local" / "auto_research_runs" / "proxy_baseline" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        if name == "proxy_baseline_eval_mmlu-redux":
            out = run_repo / "local" / "auto_research_runs" / "proxy_baseline" / "results" / "mmlu-redux"
            out.mkdir(parents=True, exist_ok=True)
            (out / "Rosetta_mmlu-redux_generate_summary.json").write_text(
                json.dumps({"model": "Rosetta", "dataset": "mmlu-redux", "answer_method": "generate", "overall_accuracy": 0.5}),
                encoding="utf-8",
            )
        if name == "proxy_command_1":
            out = run_repo / "local" / "auto_research_runs" / "idea_proxy_replay" / "proxy" / "results" / "mmlu-redux"
            out.mkdir(parents=True, exist_ok=True)
            (out / "Rosetta_mmlu-redux_generate_summary.json").write_text(
                json.dumps({"model": "Rosetta", "dataset": "mmlu-redux", "answer_method": "generate", "overall_accuracy": 0.505}),
                encoding="utf-8",
            )
            (out / "prediction_outputs.jsonl").write_text(
                "\n".join(
                    json.dumps({"prediction": f"Answer: {label}", "answer": label})
                    for label in ["A", "B", "C", "D"]
                )
                + "\n",
                encoding="utf-8",
            )
        if name == "activation_smoke_eval_mmlu-redux":
            out = run_repo / "local" / "auto_research_runs" / "idea_proxy_replay" / "proxy" / "activation_smoke_disabled" / "results" / "mmlu-redux"
            out.mkdir(parents=True, exist_ok=True)
            (out / "Rosetta_mmlu-redux_generate_summary.json").write_text(
                json.dumps({"model": "Rosetta", "dataset": "mmlu-redux", "answer_method": "generate", "overall_accuracy": 0.49}),
                encoding="utf-8",
            )
        if name == "train":
            final = run_repo / "local" / "auto_research_runs" / "idea_proxy_replay" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        if name == "eval_mmlu-redux":
            out = run_repo / "local" / "auto_research_runs" / "idea_proxy_replay" / "results" / "mmlu-redux"
            out.mkdir(parents=True, exist_ok=True)
            (out / "Rosetta_mmlu-redux_generate_summary.json").write_text(
                json.dumps({"model": "Rosetta", "dataset": "mmlu-redux", "answer_method": "generate", "overall_accuracy": 0.51}),
                encoding="utf-8",
            )
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}

    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    gpu_selection = agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1})
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={
            "id": "idea_proxy_replay",
            "title": "Idea Proxy Replay",
            "experiment_contract": {
                "ablation_switch": "disable_mechanism",
                "config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}},
            },
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=gpu_selection,
        proxy_gpu_selection=gpu_selection,
    )

    proxy = result["proxy_screen"]
    baseline_cache = paths.root / "experiment" / "results" / "c2c_proxy_baseline.json"
    assert baseline_cache.exists()
    assert seen_steps.index("proxy_baseline_train") < seen_steps.index("proxy_command_0")
    assert seen_steps.index("activation_smoke_eval_mmlu-redux") < seen_steps.index("train")
    assert seen_steps.index("proxy_command_1") < seen_steps.index("train")
    assert proxy["status"] == "passed"
    assert proxy["proxy_delta_vs_baseline"] == 0.5
    assert proxy["full_baseline_mean"] == 50.0
    assert proxy["proxy_baseline_mean"] == 50.0
    assert proxy["comparison_baseline_mean"] == 50.0
    assert proxy["proxy_delta_vs_comparison_baseline"] == 0.5
    assert proxy["proxy_delta_vs_proxy_baseline"] == 0.5
    assert proxy["artifact_paths"]["proxy_metrics"].endswith("proxy_metrics.json")
    assert proxy["artifact_paths"]["proxy_baseline_metrics"].endswith("c2c_proxy_baseline.json")
    assert proxy["proxy_decision_mode"] == "paired_baseline"
    assert proxy["eval_smoke"]["status"] == "ok"
    assert proxy["eval_smoke"]["answer_parse_rate"] == 1.0
    assert result["activation_smoke"]["status"] == "passed"
    assert result["activation_smoke"]["comparison"]["enabled_minus_disabled_mean"] == 1.5
    readiness = result["full_s3_readiness"]
    assert readiness["status"] == "ready"
    assert readiness["full_train_allowed"] is True
    assert readiness["static_risk"]["status"] == "clean"
    assert readiness["proxy"]["delta"] == 0.5
    assert readiness["eval_smoke"]["healthy"] is True
    assert readiness["activation_smoke"]["status"] == "passed"
    assert readiness["activation_smoke"]["no_op"] is False
    assert readiness["ablation_switch"]["declared"] is True
    assert readiness["worth_full_train"]["decision"] == "yes"
    assert "experiment/results/full_s3_readiness_report.json" in readiness["artifact_paths"]["project_readiness_report"]
    assert (paths.root / "experiment" / "results" / "full_s3_readiness_report.json").exists()
    assert result["command_status"] == "ok"


def test_c2c_proxy_baseline_eval_timeout_uses_configured_fallback(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "replay",
                "eval_datasets": ["mmlu-redux"],
                "eval_limit": 2,
                "train_samples": 2,
                "require_proxy_metrics": True,
                "require_paired_baseline": True,
                "run_baseline_if_missing": True,
                "allow_configured_baseline_fallback": True,
                "min_proxy_mean_delta": -0.3,
                "max_proxy_dataset_regression": 1.5,
                "activation_smoke": {"enabled": False},
            },
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_proxy_baseline_timeout_fallback", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    seen_steps = []

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        del command, retry_policy
        run_repo = Path(working_dir)
        seen_steps.append(name)
        if name == "proxy_baseline_train":
            final = run_repo / "local" / "auto_research_runs" / "proxy_baseline" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        if name == "proxy_baseline_eval_mmlu-redux":
            return {
                "step": name,
                "status": "failed",
                "returncode": 124,
                "attempts": [
                    {
                        "stdout": "baseline eval started",
                        "stderr": "Command timed out after 1200s",
                        "timed_out": True,
                        "elapsed_seconds": 1200.0,
                        "timeout_seconds": 1200,
                    }
                ],
            }
        if name == "proxy_command_0":
            final = run_repo / "local" / "auto_research_runs" / "idea_proxy_timeout" / "proxy" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        if name == "proxy_command_1":
            out = run_repo / "local" / "auto_research_runs" / "idea_proxy_timeout" / "proxy" / "results" / "mmlu-redux"
            out.mkdir(parents=True, exist_ok=True)
            (out / "Rosetta_mmlu-redux_generate_summary.json").write_text(
                json.dumps({"model": "Rosetta", "dataset": "mmlu-redux", "answer_method": "generate", "overall_accuracy": 0.505}),
                encoding="utf-8",
            )
            (out / "prediction_outputs.jsonl").write_text(
                "\n".join(json.dumps({"prediction": f"Answer: {label}", "answer": label}) for label in ["A", "B", "C", "D"]) + "\n",
                encoding="utf-8",
            )
        if name == "train":
            final = run_repo / "local" / "auto_research_runs" / "idea_proxy_timeout" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        if name == "eval_mmlu-redux":
            out = run_repo / "local" / "auto_research_runs" / "idea_proxy_timeout" / "results" / "mmlu-redux"
            out.mkdir(parents=True, exist_ok=True)
            (out / "Rosetta_mmlu-redux_generate_summary.json").write_text(
                json.dumps({"model": "Rosetta", "dataset": "mmlu-redux", "answer_method": "generate", "overall_accuracy": 0.51}),
                encoding="utf-8",
            )
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}

    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    gpu_selection = agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1})
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={
            "id": "idea_proxy_timeout",
            "title": "Idea Proxy Timeout",
            "experiment_contract": {
                "config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}},
            },
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=gpu_selection,
        proxy_gpu_selection=gpu_selection,
    )

    state = json.loads(Path(result["run_state_path"]).read_text(encoding="utf-8"))
    command_log = json.loads((paths.root / "experiment" / "logs" / "c2c_idea_proxy_timeout_commands.json").read_text(encoding="utf-8"))
    proxy = result["proxy_screen"]

    assert "proxy_baseline_eval_mmlu-redux" in seen_steps
    assert "proxy_command_0" in seen_steps
    assert state["proxy_baseline"]["status"] == "fallback"
    assert state["proxy_baseline"]["command_failure"]["category"] == "proxy_timeout"
    assert proxy["status"] == "passed"
    assert proxy["proxy_baseline"]["source"] == "configured_full_baseline_subset_fallback"
    assert proxy["proxy_delta_vs_proxy_baseline"] == 0.5
    assert result["command_status"] == "ok"
    baseline_eval_run = next(item for item in command_log["runs"] if item["step"] == "proxy_baseline_eval_mmlu-redux")
    assert baseline_eval_run["returncode"] == 124


def test_c2c_proxy_activation_smoke_blocks_no_effect_before_full_training(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "replay",
                "eval_datasets": ["mmlu-redux"],
                "eval_limit": 2,
                "train_samples": 2,
                "require_proxy_metrics": True,
                "require_paired_baseline": True,
                "run_baseline_if_missing": True,
                "min_proxy_mean_delta": -0.3,
                "max_proxy_dataset_regression": 1.5,
                "activation_smoke": {"enabled": True, "max_datasets": 1, "min_abs_metric_delta": 0.01},
            },
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_proxy_activation_no_effect", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    seen_steps: list[str] = []

    def write_summary(root: Path, dataset: str, accuracy: float) -> None:
        out = root / dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / f"Rosetta_{dataset}_generate_summary.json").write_text(
            json.dumps({"model": "Rosetta", "dataset": dataset, "answer_method": "generate", "overall_accuracy": accuracy}),
            encoding="utf-8",
        )

    def write_predictions(root: Path, dataset: str, labels: list[str]) -> None:
        out = root / dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / "prediction_outputs.jsonl").write_text(
            "\n".join(json.dumps({"prediction": f"Answer: {label}", "answer": label}) for label in labels) + "\n",
            encoding="utf-8",
        )

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        del command, retry_policy
        run_repo = Path(working_dir)
        seen_steps.append(name)
        if name == "proxy_baseline_train":
            final = run_repo / "local" / "auto_research_runs" / "proxy_baseline" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        elif name == "proxy_baseline_eval_mmlu-redux":
            write_summary(run_repo / "local" / "auto_research_runs" / "proxy_baseline" / "results", "mmlu-redux", 0.50)
        elif name == "proxy_command_0":
            final = run_repo / "local" / "auto_research_runs" / "no_effect" / "proxy" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        elif name == "proxy_command_1":
            root = run_repo / "local" / "auto_research_runs" / "no_effect" / "proxy" / "results"
            write_summary(root, "mmlu-redux", 0.505)
            write_predictions(root, "mmlu-redux", ["A", "B", "C", "D"])
        elif name == "activation_smoke_eval_mmlu-redux":
            root = run_repo / "local" / "auto_research_runs" / "no_effect" / "proxy" / "activation_smoke_disabled" / "results"
            write_summary(root, "mmlu-redux", 0.505)
            write_predictions(root, "mmlu-redux", ["A", "B", "C", "D"])
        elif name == "train" or name.startswith("eval_"):
            raise AssertionError("activation smoke no-effect must not reach full train/eval")
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}

    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={
            "id": "no_effect",
            "title": "No Effect",
            "experiment_contract": {
                "ablation_switch": "disable_mechanism",
                "config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}},
            },
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
        proxy_gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    state = json.loads(Path(result["run_state_path"]).read_text(encoding="utf-8"))
    assert "activation_smoke_eval_mmlu-redux" in seen_steps
    assert "train" not in seen_steps
    assert result["decision"] == "proxy_repairable"
    assert result["command_status"] == "proxy_repairable"
    assert result["activation_smoke"]["status"] == "failed"
    assert result["activation_smoke"]["comparison"]["enabled_minus_disabled_mean"] == 0.0
    assert result["activation_smoke"]["comparison"]["prediction_comparison"]["prediction_diff_rate"] == 0.0
    assert result["proxy_screen"]["status"] == "repairable_proxy_risk"
    assert result["failure_attribution"]["primary_failure"] == "proxy_activation_smoke_no_effect"
    assert state["train"] is None


def test_c2c_full_s3_readiness_blocks_train_when_not_ready(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "replay",
                "eval_datasets": ["mmlu-redux"],
                "eval_limit": 2,
                "train_samples": 2,
                "require_proxy_metrics": True,
                "require_paired_baseline": True,
                "run_baseline_if_missing": True,
                "min_proxy_mean_delta": -0.3,
                "max_proxy_dataset_regression": 1.5,
                "activation_smoke": {
                    "enabled": True,
                    "hard_gate": False,
                    "max_datasets": 1,
                    "min_abs_metric_delta": 0.01,
                },
            },
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_full_readiness_blocks_train", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    seen_steps: list[str] = []

    def write_summary(root: Path, dataset: str, accuracy: float) -> None:
        out = root / dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / f"Rosetta_{dataset}_generate_summary.json").write_text(
            json.dumps({"model": "Rosetta", "dataset": dataset, "answer_method": "generate", "overall_accuracy": accuracy}),
            encoding="utf-8",
        )

    def write_predictions(root: Path, dataset: str, labels: list[str]) -> None:
        out = root / dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / "prediction_outputs.jsonl").write_text(
            "\n".join(json.dumps({"prediction": f"Answer: {label}", "answer": label}) for label in labels) + "\n",
            encoding="utf-8",
        )

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        del command, retry_policy
        run_repo = Path(working_dir)
        seen_steps.append(name)
        if name == "proxy_baseline_train":
            final = run_repo / "local" / "auto_research_runs" / "proxy_baseline" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        elif name == "proxy_baseline_eval_mmlu-redux":
            write_summary(run_repo / "local" / "auto_research_runs" / "proxy_baseline" / "results", "mmlu-redux", 0.50)
        elif name == "proxy_command_0":
            final = run_repo / "local" / "auto_research_runs" / "readiness_not_ready" / "proxy" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        elif name == "proxy_command_1":
            root = run_repo / "local" / "auto_research_runs" / "readiness_not_ready" / "proxy" / "results"
            write_summary(root, "mmlu-redux", 0.505)
            write_predictions(root, "mmlu-redux", ["A", "B", "C", "D"])
        elif name == "activation_smoke_eval_mmlu-redux":
            root = run_repo / "local" / "auto_research_runs" / "readiness_not_ready" / "proxy" / "activation_smoke_disabled" / "results"
            write_summary(root, "mmlu-redux", 0.505)
            write_predictions(root, "mmlu-redux", ["A", "B", "C", "D"])
        elif name == "train" or name.startswith("eval_") or name.startswith("ablation_eval_"):
            raise AssertionError("full_s3_readiness not_ready must not reach full train/eval/ablation")
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}

    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={
            "id": "readiness_not_ready",
            "title": "Readiness Not Ready",
            "experiment_contract": {
                "ablation_switch": "disable_mechanism",
                "config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}},
            },
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
        proxy_gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    state = json.loads(Path(result["run_state_path"]).read_text(encoding="utf-8"))
    readiness = result["full_s3_readiness"]
    assert "activation_smoke_eval_mmlu-redux" in seen_steps
    assert "train" not in seen_steps
    assert not any(step.startswith("eval_") for step in seen_steps)
    assert readiness["status"] == "not_ready"
    assert readiness["full_train_allowed"] is False
    assert result["decision"] == "proxy_repairable"
    assert result["command_status"] == "proxy_repairable"
    assert result["proxy_screen"]["status"] == "repairable_proxy_risk"
    assert result["proxy_screen"]["repair_mode"] == "full_s3_readiness_repair"
    assert result["proxy_screen"]["proxy_effect_repair_contract"]["source"] == "full_s3_readiness"
    assert result["failure_attribution"]["primary_failure"] == "full_s3_readiness_not_ready"
    assert state["train"] is None
    assert any(action["action"] == "block_full_train_until_readiness" for action in state["recovery_actions"])


def test_c2c_proxy_activation_smoke_allows_metric_neutral_when_wiring_trace_passed(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "replay",
                "eval_datasets": ["mmlu-redux"],
                "eval_limit": 2,
                "train_samples": 2,
                "require_proxy_metrics": True,
                "require_paired_baseline": True,
                "run_baseline_if_missing": True,
                "min_proxy_mean_delta": -0.3,
                "max_proxy_dataset_regression": 1.5,
                "activation_smoke": {"enabled": True, "max_datasets": 1, "min_abs_metric_delta": 0.01},
            },
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_proxy_activation_wired_metric_neutral", simulate=False)
    validation_path = paths.root / "plan" / "code_patches" / "wired" / "validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "name": "runtime_smoke:mechanism_activation_wiring",
                        "status": "ok",
                        "returncode": 0,
                        "switch": "disable_mechanism",
                        "runtime_code_refs": {"switch_refs": ["rosetta/model/projector.py"]},
                        "rosetta_config": {"disabled_switch_value": True},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    seen_steps: list[str] = []

    def write_summary(root: Path, dataset: str, accuracy: float) -> None:
        out = root / dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / f"Rosetta_{dataset}_generate_summary.json").write_text(
            json.dumps({"model": "Rosetta", "dataset": dataset, "answer_method": "generate", "overall_accuracy": accuracy}),
            encoding="utf-8",
        )

    def write_predictions(root: Path, dataset: str, labels: list[str]) -> None:
        out = root / dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / "prediction_outputs.jsonl").write_text(
            "\n".join(json.dumps({"prediction": f"Answer: {label}", "answer": label}) for label in labels) + "\n",
            encoding="utf-8",
        )

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        del command, retry_policy
        run_repo = Path(working_dir)
        seen_steps.append(name)
        if name == "proxy_baseline_train":
            final = run_repo / "local" / "auto_research_runs" / "proxy_baseline" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        elif name == "proxy_baseline_eval_mmlu-redux":
            write_summary(run_repo / "local" / "auto_research_runs" / "proxy_baseline" / "results", "mmlu-redux", 0.50)
        elif name == "proxy_command_0":
            final = run_repo / "local" / "auto_research_runs" / "wired_neutral" / "proxy" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        elif name == "proxy_command_1":
            root = run_repo / "local" / "auto_research_runs" / "wired_neutral" / "proxy" / "results"
            write_summary(root, "mmlu-redux", 0.505)
            write_predictions(root, "mmlu-redux", ["A", "B", "C", "D"])
        elif name == "activation_smoke_eval_mmlu-redux":
            root = run_repo / "local" / "auto_research_runs" / "wired_neutral" / "proxy" / "activation_smoke_disabled" / "results"
            write_summary(root, "mmlu-redux", 0.505)
            write_predictions(root, "mmlu-redux", ["A", "B", "C", "D"])
        elif name == "train":
            final = run_repo / "local" / "auto_research_runs" / "wired_neutral" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        elif name == "eval_mmlu-redux":
            write_summary(run_repo / "local" / "auto_research_runs" / "wired_neutral" / "results", "mmlu-redux", 0.51)
        elif name == "ablation_eval_mmlu-redux":
            write_summary(run_repo / "local" / "auto_research_runs" / "wired_neutral" / "ablation_disabled" / "results", "mmlu-redux", 0.50)
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}

    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={
            "id": "wired_neutral",
            "title": "Wired Neutral",
            "code_patch": {"status": "ok", "validation": validation_path.relative_to(paths.root).as_posix()},
            "experiment_contract": {
                "ablation_switch": "disable_mechanism",
                "config_overrides": {"eval": {"model": {"rosetta_config": {"mechanism_enabled": True}}}},
            },
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
        proxy_gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    comparison = result["activation_smoke"]["comparison"]
    assert result["activation_smoke"]["status"] == "passed"
    assert comparison["mechanism_observed"] is False
    assert comparison["mechanism_wired_metric_neutral"] is True
    assert result["activation_smoke"]["mechanism_trace"]["status"] == "wired"
    assert "train" in seen_steps
    assert result["command_status"] == "ok"


def test_c2c_proxy_activation_smoke_passes_metric_neutral_prediction_change(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "replay",
                "eval_datasets": ["mmlu-redux"],
                "eval_limit": 4,
                "train_samples": 2,
                "require_proxy_metrics": True,
                "require_paired_baseline": True,
                "run_baseline_if_missing": True,
                "min_proxy_mean_delta": -0.3,
                "max_proxy_dataset_regression": 1.5,
                "activation_smoke": {
                    "enabled": True,
                    "max_datasets": 1,
                    "min_abs_metric_delta": 0.01,
                    "min_prediction_diff_rate": 0.25,
                },
            },
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_proxy_activation_prediction_change", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    seen_steps: list[str] = []

    def write_summary_and_predictions(root: Path, dataset: str, accuracy: float, labels: list[str]) -> None:
        out = root / dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / f"Rosetta_{dataset}_generate_summary.json").write_text(
            json.dumps({"model": "Rosetta", "dataset": dataset, "answer_method": "generate", "overall_accuracy": accuracy}),
            encoding="utf-8",
        )
        (out / "prediction_outputs.jsonl").write_text(
            "\n".join(json.dumps({"prediction": f"Answer: {label}", "answer": label}) for label in labels) + "\n",
            encoding="utf-8",
        )

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        del command, retry_policy
        run_repo = Path(working_dir)
        seen_steps.append(name)
        if name == "proxy_baseline_train":
            final = run_repo / "local" / "auto_research_runs" / "proxy_baseline" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        elif name == "proxy_baseline_eval_mmlu-redux":
            write_summary_and_predictions(run_repo / "local" / "auto_research_runs" / "proxy_baseline" / "results", "mmlu-redux", 0.50, ["A", "B", "C", "D"])
        elif name == "proxy_command_0":
            final = run_repo / "local" / "auto_research_runs" / "prediction_change" / "proxy" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        elif name == "proxy_command_1":
            write_summary_and_predictions(run_repo / "local" / "auto_research_runs" / "prediction_change" / "proxy" / "results", "mmlu-redux", 0.505, ["A", "B", "C", "D"])
        elif name == "activation_smoke_eval_mmlu-redux":
            write_summary_and_predictions(
                run_repo / "local" / "auto_research_runs" / "prediction_change" / "proxy" / "activation_smoke_disabled" / "results",
                "mmlu-redux",
                0.505,
                ["A", "A", "C", "D"],
            )
        elif name == "train":
            final = run_repo / "local" / "auto_research_runs" / "prediction_change" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        elif name == "eval_mmlu-redux":
            write_summary_and_predictions(run_repo / "local" / "auto_research_runs" / "prediction_change" / "results", "mmlu-redux", 0.51, ["A", "B", "C", "D"])
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}

    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={
            "id": "prediction_change",
            "title": "Prediction Change",
            "experiment_contract": {
                "ablation_switch": "disable_mechanism",
                "config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}},
            },
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
        proxy_gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    comparison = result["activation_smoke"]["comparison"]
    assert seen_steps.index("activation_smoke_eval_mmlu-redux") < seen_steps.index("train")
    assert result["activation_smoke"]["status"] == "passed"
    assert comparison["enabled_minus_disabled_mean"] == 0.0
    assert comparison["prediction_comparison"]["prediction_diff_rate"] == 0.25
    assert comparison["prediction_comparison"]["answer_diff_rate"] == 0.25
    assert comparison["mechanism_observed"] is True
    assert result["command_status"] == "ok"


def test_c2c_eval_smoke_detects_all_zero_empty_outputs(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    out = result_root / "mmlu-redux"
    out.mkdir(parents=True)
    (out / "Rosetta_mmlu-redux_generate_summary.json").write_text(
        json.dumps({"model": "Rosetta", "dataset": "mmlu-redux", "answer_method": "generate", "overall_accuracy": 0.0}),
        encoding="utf-8",
    )
    (out / "prediction_outputs.jsonl").write_text(
        "\n".join(json.dumps({"prediction": "", "answer": "A"}) for _ in range(4)) + "\n",
        encoding="utf-8",
    )

    smoke = collect_c2c_eval_smoke(result_root, repo_root=tmp_path)

    assert smoke["status"] == "warning"
    assert smoke["summary_datasets"]["mmlu-redux"]["accuracy_percent"] == 0.0
    assert smoke["sample_count"] == 4
    assert smoke["nonempty_prediction_rate"] == 0.0
    assert "all_summary_scores_zero" in smoke["red_flags"]
    assert "low_nonempty_prediction_rate" in smoke["red_flags"]


def test_c2c_proxy_all_zero_records_eval_smoke_failure(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "replay",
                "eval_datasets": ["mmlu-redux"],
                "eval_limit": 2,
                "train_samples": 2,
                "require_proxy_metrics": True,
                "require_paired_baseline": True,
                "run_baseline_if_missing": True,
                "min_proxy_mean_delta": -0.3,
                "max_proxy_dataset_regression": 1.5,
                "eval_smoke": {"enabled": True, "min_nonempty_prediction_rate": 0.5, "min_answer_parse_rate": 0.2},
            },
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_proxy_all_zero_smoke", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        run_repo = Path(working_dir)
        if name == "proxy_baseline_train":
            final = run_repo / "local" / "auto_research_runs" / "proxy_baseline" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        if name == "proxy_baseline_eval_mmlu-redux":
            out = run_repo / "local" / "auto_research_runs" / "proxy_baseline" / "results" / "mmlu-redux"
            out.mkdir(parents=True, exist_ok=True)
            (out / "Rosetta_mmlu-redux_generate_summary.json").write_text(
                json.dumps({"model": "Rosetta", "dataset": "mmlu-redux", "answer_method": "generate", "overall_accuracy": 0.5}),
                encoding="utf-8",
            )
        if name == "proxy_command_1":
            out = run_repo / "local" / "auto_research_runs" / "all_zero_proxy" / "proxy" / "results" / "mmlu-redux"
            out.mkdir(parents=True, exist_ok=True)
            (out / "Rosetta_mmlu-redux_generate_summary.json").write_text(
                json.dumps({"model": "Rosetta", "dataset": "mmlu-redux", "answer_method": "generate", "overall_accuracy": 0.0}),
                encoding="utf-8",
            )
            (out / "prediction_outputs.jsonl").write_text(
                "\n".join(json.dumps({"prediction": "", "answer": "A"}) for _ in range(4)) + "\n",
                encoding="utf-8",
            )
        if name == "train" or name.startswith("eval_"):
            raise AssertionError("hard proxy reject must not reach full train/eval")
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}

    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={
            "id": "all_zero_proxy",
            "title": "All Zero Proxy",
            "experiment_contract": {"config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}}},
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
        proxy_gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    proxy = result["proxy_screen"]
    assert result["decision"] == "proxy_rejected"
    assert proxy["status"] == "rejected"
    assert proxy["metrics"]["mean"] == 0.0
    assert proxy["eval_smoke"]["status"] == "warning"
    assert "all_summary_scores_zero" in proxy["eval_smoke"]["red_flags"]
    assert proxy["proxy_eval_health_failure"]["status"] == "suspected_output_or_parser_failure"
    assert result["failure_attribution"]["primary_failure"] == "proxy_eval_output_health_failure"
    assert result["failure_attribution"]["proxy_eval_health_failure"]["red_flags"]


def test_c2c_failure_attribution_records_dataset_sample_and_patch_risk() -> None:
    candidate = {
        "id": "mixed_tradeoff",
        "decision": "not_viable",
        "metrics": {"mean": 50.0, "datasets": {"mmlu-redux": 43.0, "ai2-arc": 55.0, "openbookqa": 52.0}},
        "delta_vs_baseline": -0.82,
        "patch_result": {"changed_files": ["rosetta/model/projector.py", "script/evaluation/unified_evaluator.py"]},
        "config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}},
    }
    baseline = {"mean": 50.82, "datasets": {"mmlu-redux": 47.0, "ai2-arc": 54.0, "openbookqa": 50.0}}

    attribution = ExperimentAgent._c2c_failure_attribution(candidate, baseline)

    assert attribution["primary_failure"] == "mmlu-redux_regression"
    assert attribution["dragging_datasets"][0]["dataset"] == "mmlu-redux"
    assert attribution["sample_type_failures"][0]["sample_family"] == "multi_domain_knowledge_reasoning"
    assert "openbookqa_gain_mmlu_redux_regression" in attribution["mixed_gain_patterns"]
    assert "evaluation_code_changed" in attribution["patch_risk"]["risk_labels"]
    assert "projector_mechanism_changed" in attribution["patch_risk"]["risk_labels"]
    assert "train.model.soft_alignment_top_k" in attribution["patch_risk"]["config_override_keys"]


def test_c2c_failure_attribution_uses_proxy_dataset_deltas_without_full_metrics() -> None:
    candidate = {
        "id": "proxy_tradeoff",
        "decision": "proxy_repairable",
        "metrics": {},
        "proxy_screen": {
            "status": "repairable_proxy_risk",
            "metrics": {"mean": 32.6, "datasets": {"ai2-arc": 32.5, "mmlu-redux": 32.0, "openbookqa": 33.3}},
            "proxy_baseline": {"mean": 32.0, "datasets": {"ai2-arc": 33.8, "mmlu-redux": 30.7, "openbookqa": 31.5}},
            "proxy_dataset_deltas": {"ai2-arc": -1.3, "mmlu-redux": 1.3, "openbookqa": 1.8},
        },
    }

    attribution = ExperimentAgent._c2c_failure_attribution(candidate, {"mean": 50.0, "datasets": {}})

    assert attribution["primary_failure"] == "repairable_proxy_risk_before_full_training"
    assert attribution["dragging_datasets"][0]["dataset"] == "ai2-arc"
    assert attribution["dragging_datasets"][0]["source"] == "proxy_screen"
    assert attribution["sample_type_failures"][0]["sample_family"] == "science_reasoning_challenge"
    assert "cross_dataset_tradeoff" in attribution["mixed_gain_patterns"]


def test_c2c_ablation_no_effect_is_failure_attribution() -> None:
    candidate = {
        "id": "noop_mechanism",
        "decision": "not_viable",
        "metrics": {"mean": 55.0, "datasets": {"mmlu-redux": 55.0}},
        "delta_vs_baseline": 5.0,
        "ablation": {
            "enabled": True,
            "status": "ok",
            "switch": "disable_noop",
            "comparison": {
                "status": "ok",
                "enabled_minus_disabled_mean": 0.0,
                "dataset_enabled_minus_disabled": {"mmlu-redux": 0.0},
                "mechanism_supported": False,
            },
        },
    }
    baseline = {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}}

    attribution = ExperimentAgent._c2c_failure_attribution(candidate, baseline)
    posthoc = ExperimentAgent._c2c_deterministic_posthoc_review(
        {
            "acceptance": {"passed": False, "reason": "mechanism ablation support not met"},
            "baseline": baseline,
            "best_candidate": candidate,
            "candidate_results": [candidate],
        }
    )

    assert attribution["primary_failure"] == "ablation_no_effect"
    assert attribution["ablation_evidence"]["status"] == "no_effect"
    assert any("ablation switch did not change metrics" in item for item in posthoc["failure_modes"])


def test_c2c_acceptance_requires_ablation_support_when_configured() -> None:
    best = {
        "decision": "not_viable",
        "metrics": {"mean": 55.0, "datasets": {"mmlu-redux": 55.0}},
        "worst_dataset_regression": 0.0,
        "acceptance_rule": {"require_ablation_support": True},
        "mechanism_supported": False,
    }

    comparison = ExperimentAgent._c2c_acceptance_comparison(
        best,
        {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        min_delta=0.1,
        max_regression=2.0,
    )

    assert comparison["passed"] is False
    assert comparison["reason"] == "mechanism ablation support not met"


def test_c2c_acceptance_defaults_to_effect_first_without_ablation_support() -> None:
    best = {
        "decision": "not_viable",
        "metrics": {"mean": 55.0, "datasets": {"mmlu-redux": 55.0}},
        "worst_dataset_regression": 0.0,
        "acceptance_rule": {"require_ablation_support": False},
        "mechanism_supported": False,
    }

    comparison = ExperimentAgent._c2c_acceptance_comparison(
        best,
        {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        min_delta=0.1,
        max_regression=2.0,
    )

    assert comparison["passed"] is True
    assert comparison["reason"] == "accepted"
    assert comparison["require_ablation_support"] is False


def test_c2c_proxy_calibration_marks_false_positive_and_dataset_errors() -> None:
    payload = {
        "baseline": {"mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "acceptance": {"passed": False, "reason": "mean delta or dataset regression threshold not met"},
        "candidate_results": [
            {
                "id": "utility_proxy_pass_full_fail",
                "title": "Utility proxy pass full fail",
                "mechanism_type": "utility_predicted_cache_routing",
                "decision": "not_viable",
                "metrics": {"mean": 49.8, "datasets": {"mmlu-redux": 48.5, "ai2-arc": 50.2, "openbookqa": 50.7}},
                "delta_vs_baseline": -0.2,
                "proxy_screen": {
                    "status": "passed",
                    "proxy_delta_vs_baseline": 0.8,
                    "proxy_score": 0.6,
                    "proxy_dataset_deltas": {"mmlu-redux": 0.9, "ai2-arc": 0.2, "openbookqa": 0.3},
                    "metrics": {"mean": 50.8, "datasets": {"mmlu-redux": 50.9, "ai2-arc": 50.2, "openbookqa": 50.3}},
                },
            },
            {
                "id": "semantic_proxy_pass_full_win",
                "title": "Semantic proxy pass full win",
                "mechanism_type": "semantic_span_graph_alignment",
                "decision": "candidate_win",
                "metrics": {"mean": 51.0, "datasets": {"mmlu-redux": 51.0, "ai2-arc": 51.0, "openbookqa": 51.0}},
                "delta_vs_baseline": 1.0,
                "proxy_screen": {
                    "status": "passed",
                    "proxy_delta_vs_baseline": 0.5,
                    "proxy_score": 0.7,
                    "proxy_dataset_deltas": {"mmlu-redux": 0.4, "ai2-arc": 0.6, "openbookqa": 0.5},
                    "metrics": {"mean": 50.5, "datasets": {"mmlu-redux": 50.4, "ai2-arc": 50.6, "openbookqa": 50.5}},
                },
            }
        ],
    }

    iteration = experiment_module._c2c_proxy_calibration_iteration(payload, iteration=3)
    summary = experiment_module._c2c_proxy_calibration_summary([iteration])

    candidate = iteration["candidates"][0]
    assert iteration["full_s3_completed_candidate_count"] == 2
    assert candidate["proxy_false_positive"] is True
    assert candidate["false_positive_reason"] == "proxy_mean_positive_full_mean_nonpositive"
    assert candidate["mispredicted_datasets"] == ["mmlu-redux"]
    assert candidate["dataset_calibration"]["mmlu-redux"]["proxy_delta"] == 0.9
    assert candidate["dataset_calibration"]["mmlu-redux"]["full_delta"] == -1.5
    assert summary["proxy_false_positive_rate"] == 0.5
    assert summary["proxy_full_delta_correlation"] is not None
    assert summary["dataset_error_summary"]["mmlu-redux"]["misprediction_count"] == 1
    assert summary["mechanism_false_positive_summary"]["utility_predicted_cache_routing"]["false_positive_rate"] == 1.0
    assert summary["mechanism_false_positive_summary"]["utility_predicted_cache_routing"]["proxy_positive_full_nonpositive_rate"] == 1.0
    assert summary["mechanism_false_positive_summary"]["utility_predicted_cache_routing"]["mispredicted_datasets"]["mmlu-redux"] == 1
    method_feedback = summary["method_feedback"]
    assert method_feedback["risky_datasets"][0]["dataset"] == "mmlu-redux"
    assert method_feedback["risky_mechanisms"][0]["mechanism_type"] == "utility_predicted_cache_routing"
    assert any("risky datasets" in item for item in method_feedback["recommendations"])


def test_c2c_paperization_readiness_after_effect_win() -> None:
    readiness = experiment_module._c2c_paperization_readiness(
        {
            "id": "winner",
            "proxy_screen": {
                "quality_repair": {
                    "issues": ["missing_coverage_diagnostics_evidence", "missing_matched_coverage_evidence"]
                }
            },
            "patch_result": {
                "mechanism_review": {
                    "soft_issues": ["ablation_switch_not_wired"]
                }
            },
        },
        {"passed": True},
    )

    assert readiness["status"] == "ready"
    assert readiness["next_stage"] == "paperization"
    assert readiness["candidate_id"] == "winner"
    assert any("coverage diagnostics" in item for item in readiness["tasks"])
    assert any("ablation switch" in item for item in readiness["tasks"])


def test_disable_llm_during_execution_skips_patch_llm(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["experiment"]["disable_llm_during_execution"] = True
    context = AgentContext(tmp_path / "workspace" / "p", config, ArtifactManager(tmp_path / "workspace" / "p"), ModelClient(config, project_root=tmp_path))
    agent = ExperimentAgent(context)

    class BombLLM:
        use_real_api = True

        def generate_json(self, **kwargs):
            raise AssertionError("training execution must not call LLM patch generation")

    context.llm = BombLLM()
    adapter = C2CAdapter(tmp_path, {"c2c": {"snapshot_path": str(_fake_c2c_repo(tmp_path)), "env_python": "/usr/bin/python3"}})
    patch = agent._generate_c2c_patch({"id": "x"}, adapter)
    assert patch["operations"] == []
    assert patch["status"] == "missing"
    assert "frozen S2.5 patch" in patch["summary"]


def test_c2c_posthoc_review_degrades_to_deterministic_feedback(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    paths = init_workspace(config, "topic", project_id="proj_posthoc", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    class FailingLLM:
        use_real_api = True

        def generate_json_with_schema(self, **kwargs):
            raise RuntimeError("429 Too Many Requests")

    context.llm = FailingLLM()
    payload = {
        "baseline": {"mean": 50.82, "datasets": {"mmlu-redux": 47.07, "ai2-arc": 54.78, "openbookqa": 50.6}},
        "acceptance": {
            "passed": False,
            "baseline_mean": 50.82,
            "best_mean": 50.0666,
            "delta": -0.7534,
            "min_delta_to_pass": 0.1,
            "max_dataset_regression": 2.0,
            "reason": "mean delta or dataset regression threshold not met",
        },
        "best_candidate": {
            "id": "mmlu_safe_low_confidence_gate",
            "title": "MMLU-safe low-confidence transfer gate",
            "decision": "not_viable",
            "metrics": {"mean": 50.0666, "datasets": {"mmlu-redux": 45.3303, "ai2-arc": 54.8696, "openbookqa": 50.0}},
            "dataset_regressions": {"mmlu-redux": 1.7397, "ai2-arc": 0.0, "openbookqa": 0.6},
        },
        "candidate_results": [
            {
                "id": "mmlu_safe_low_confidence_gate",
                "title": "MMLU-safe low-confidence transfer gate",
                "decision": "not_viable",
                "metrics": {"mean": 50.0666, "datasets": {"mmlu-redux": 45.3303}},
                "dataset_regressions": {"mmlu-redux": 1.7397},
            }
        ],
    }

    review = agent._c2c_posthoc_review(payload)

    assert review["status"] == "degraded"
    assert "GPT posthoc review unavailable" in review["reason"]
    assert review["failure_modes"]
    assert review["next_round_suggestions"]
    assert review["avoid_repeat_rules"]
    assert review["feedback_entries"][0]["idea_id"] == "mmlu_safe_low_confidence_gate"


def test_c2c_posthoc_review_uses_deterministic_feedback_without_llm(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    paths = init_workspace(config, "topic", project_id="proj_posthoc_no_llm", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    class NoLLM:
        use_real_api = False

    context.llm = NoLLM()
    payload = {
        "baseline": {"mean": 50.82, "datasets": {"mmlu-redux": 47.07}},
        "acceptance": {
            "passed": False,
            "baseline_mean": 50.82,
            "best_mean": 48.0,
            "delta": -2.82,
            "min_delta_to_pass": 0.1,
            "max_dataset_regression": 2.0,
            "reason": "mean delta or dataset regression threshold not met",
        },
        "best_candidate": {
            "id": "weak_gate",
            "title": "Weak gate",
            "decision": "not_viable",
            "metrics": {"mean": 48.0, "datasets": {"mmlu-redux": 43.0}},
            "dataset_regressions": {"mmlu-redux": 4.07},
        },
        "candidate_results": [
            {
                "id": "weak_gate",
                "title": "Weak gate",
                "decision": "not_viable",
                "metrics": {"mean": 48.0, "datasets": {"mmlu-redux": 43.0}},
                "dataset_regressions": {"mmlu-redux": 4.07},
            }
        ],
    }

    review = agent._c2c_posthoc_review(payload)

    assert review["status"] == "deterministic_no_llm"
    assert review["next_round_suggestions"]
    assert review["avoid_repeat_rules"]
    assert review["feedback_entries"][0]["idea_id"] == "weak_gate"


def test_c2c_posthoc_review_respects_disable_llm_during_execution(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["experiment"]["disable_llm_during_execution"] = True
    paths = init_workspace(config, "topic", project_id="proj_posthoc_execution_no_llm", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    class BombLLM:
        use_real_api = True

        def generate_json_with_schema(self, **kwargs):
            raise AssertionError("S3 execution posthoc should not call the LLM when disabled")

    context.llm = BombLLM()
    payload = {
        "baseline": {"mean": 50.82, "datasets": {"mmlu-redux": 47.07}},
        "acceptance": {
            "passed": False,
            "baseline_mean": 50.82,
            "best_mean": 48.0,
            "delta": -2.82,
            "min_delta_to_pass": 0.1,
            "max_dataset_regression": 2.0,
            "reason": "mean delta or dataset regression threshold not met",
        },
        "best_candidate": {
            "id": "weak_gate",
            "title": "Weak gate",
            "decision": "not_viable",
            "metrics": {"mean": 48.0, "datasets": {"mmlu-redux": 43.0}},
            "dataset_regressions": {"mmlu-redux": 4.07},
        },
        "candidate_results": [
            {
                "id": "weak_gate",
                "title": "Weak gate",
                "decision": "not_viable",
                "metrics": {"mean": 48.0, "datasets": {"mmlu-redux": 43.0}},
                "dataset_regressions": {"mmlu-redux": 4.07},
            }
        ],
    }

    review = agent._c2c_posthoc_review(payload)

    assert review["status"] == "deterministic_execution_feedback"
    assert "disable_llm_during_execution=true" in review["reason"]
    assert review["next_round_suggestions"]
    assert review["avoid_repeat_rules"]
    assert review["feedback_entries"][0]["idea_id"] == "weak_gate"


def test_c2c_posthoc_review_summarizes_proxy_repairable_failures(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    paths = init_workspace(config, "topic", project_id="proj_posthoc_proxy", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    class NoLLM:
        use_real_api = False

    context.llm = NoLLM()
    payload = {
        "baseline": {"mean": 50.0, "datasets": {}},
        "acceptance": {"passed": False, "reason": "no candidate metrics"},
        "best_candidate": None,
        "candidate_results": [
            {
                "id": "proxy_tradeoff",
                "title": "Proxy tradeoff",
                "decision": "proxy_repairable",
                "proxy_screen": {
                    "status": "repairable_proxy_risk",
                    "reason": "proxy worst dataset regression",
                    "proxy_dataset_deltas": {"ai2-arc": -1.3, "openbookqa": 1.8},
                },
                "failure_attribution": {
                    "primary_failure": "repairable_proxy_risk_before_full_training",
                    "dragging_datasets": [{"dataset": "ai2-arc", "regression": 1.3}],
                    "patch_risk": {"risk_labels": ["projector_mechanism_changed"]},
                },
            }
        ],
    }

    review = agent._c2c_posthoc_review(payload)

    assert review["status"] == "deterministic_proxy_feedback"
    assert "cheap proxy blocked all candidates" in review["failure_modes"][0]
    assert "proxy_dataset_deltas" in " ".join(review["next_round_suggestions"])
    assert review["feedback_entries"][0]["reason"] == "proxy worst dataset regression"
    assert review["feedback_entries"][0]["proxy_screen"]["proxy_dataset_deltas"]["ai2-arc"] == -1.3


def test_c2c_posthoc_review_skips_llm_for_proxy_only_failures(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    paths = init_workspace(config, "topic", project_id="proj_posthoc_proxy_skip", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    class BombLLM:
        use_real_api = True

        def generate_json_with_schema(self, **kwargs):
            raise AssertionError("proxy-only posthoc should not call the LLM")

    context.llm = BombLLM()
    payload = {
        "baseline": {"mean": 50.0, "datasets": {}},
        "acceptance": {
            "passed": False,
            "reason": "proxy mean delta -1.2 below hard threshold -0.3",
            "proxy_best_mean": 48.8,
            "proxy_delta": -1.2,
        },
        "best_candidate": None,
        "best_proxy_candidate": {
            "id": "proxy_tradeoff",
            "title": "Proxy tradeoff",
            "decision": "proxy_rejected",
            "proxy_screen": {
                "status": "rejected",
                "metrics": {"mean": 48.8, "datasets": {"ai2-arc": 47.0}},
                "proxy_delta_vs_baseline": -1.2,
                "proxy_dataset_deltas": {"ai2-arc": -1.3},
                "reason": "proxy mean delta -1.2 below hard threshold -0.3",
            },
        },
        "candidate_results": [
            {
                "id": "proxy_tradeoff",
                "title": "Proxy tradeoff",
                "decision": "proxy_rejected",
                "proxy_screen": {
                    "status": "rejected",
                    "metrics": {"mean": 48.8, "datasets": {"ai2-arc": 47.0}},
                    "proxy_delta_vs_baseline": -1.2,
                    "proxy_dataset_deltas": {"ai2-arc": -1.3},
                    "reason": "proxy mean delta -1.2 below hard threshold -0.3",
                },
                "failure_attribution": {
                    "primary_failure": "cheap_proxy_rejected_before_full_training",
                    "dragging_datasets": [{"dataset": "ai2-arc", "regression": 1.3}],
                },
            }
        ],
    }

    review = agent._c2c_posthoc_review(payload)

    assert review["status"] == "deterministic_proxy_feedback"
    assert "skipped GPT posthoc" in review["reason"]
    assert review["feedback_entries"][0]["idea_id"] == "proxy_tradeoff"


def test_c2c_iteration_history_appends_best_and_consecutive_counts(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    paths = init_workspace(config, "topic", project_id="proj_history", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    registry_path = paths.root / "meta" / "registry.yaml"

    first = {
        "baseline": {"mean": 50.0},
        "acceptance": {"passed": False, "best_mean": 48.5, "delta": -1.5, "reason": "below"},
        "best_candidate": {
            "id": "idea_a",
            "title": "Idea A",
            "decision": "not_viable",
            "metrics": {"mean": 48.5, "datasets": {}},
            "delta_vs_baseline": -1.5,
        },
        "candidate_results": [{"id": "idea_a"}],
    }
    history = agent._append_c2c_iteration_history(first)
    assert history["iteration_count"] == 1
    assert history["best_candidate_id"] == "idea_a"
    assert history["consecutive_not_viable"] == 1

    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["iteration"] = 2
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    second = {
        "baseline": {"mean": 50.0},
        "acceptance": {"passed": False, "best_mean": 49.2, "delta": -0.8, "reason": "below"},
        "best_candidate": {
            "id": "idea_b",
            "title": "Idea B",
            "decision": "not_viable",
            "metrics": {"mean": 49.2, "datasets": {}},
            "delta_vs_baseline": -0.8,
        },
        "candidate_results": [{"id": "idea_b"}],
    }
    history = agent._append_c2c_iteration_history(second)

    assert history["iteration_count"] == 2
    assert history["best_candidate_id"] == "idea_b"
    assert history["best_delta_so_far"] == -0.8
    assert history["consecutive_not_viable"] == 2
    saved = json.loads((paths.root / "experiment/results/c2c_iteration_history.json").read_text(encoding="utf-8"))
    assert [item["iteration"] for item in saved["iterations"]] == [1, 2]


def test_c2c_iteration_history_records_proxy_rejected_metrics(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    paths = init_workspace(config, "topic", project_id="proj_history_proxy", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    candidate = {
        "id": "proxy_tradeoff",
        "title": "Proxy tradeoff",
        "decision": "proxy_rejected",
        "proxy_screen": {
            "status": "rejected",
            "metrics": {"mean": 48.8, "datasets": {"ai2-arc": 47.0}},
            "proxy_delta_vs_baseline": -1.2,
            "proxy_score": -1.9,
            "proxy_worst_dataset_regression": 1.3,
            "proxy_dataset_deltas": {"ai2-arc": -1.3},
        },
    }
    payload = {
        "baseline": {"mean": 50.0},
        "acceptance": {
            "passed": False,
            "baseline_mean": 50.0,
            "best_mean": None,
            "delta": None,
            "proxy_best_mean": 48.8,
            "proxy_delta": -1.2,
            "proxy_score": -1.9,
            "reason": "cheap proxy blocked candidate before full S3",
        },
        "best_candidate": None,
        "best_proxy_candidate": candidate,
        "candidate_results": [candidate],
    }

    history = agent._append_c2c_iteration_history(payload)

    assert history["best_candidate_id"] is None
    assert history["best_proxy_candidate_id"] == "proxy_tradeoff"
    assert history["best_proxy_mean_so_far"] == 48.8
    assert history["best_proxy_delta_so_far"] == -1.2
    saved = json.loads((paths.root / "experiment/results/c2c_iteration_history.json").read_text(encoding="utf-8"))
    entry = saved["iterations"][0]
    assert entry["acceptance"]["proxy_best_mean"] == 48.8
    assert entry["best_proxy_candidate"]["proxy_dataset_deltas"]["ai2-arc"] == -1.3
