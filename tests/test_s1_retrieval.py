import json

from auto_research.s1_retrieval import (
    default_c2c_evidence_request_plan,
    retrieve_s1_c2c_requested_evidence,
    validate_c2c_evidence_request_plan,
)


def _chunks():
    paper_chunks = [
        {"chunk_id": "paper:utility", "source_type": "paper", "source_path": "intake/c2c/paper_chunks.jsonl", "text": "utility cache transfer improves cache routing", "keywords": ["utility", "cache"]},
        {"chunk_id": "paper:coverage", "source_type": "paper", "source_path": "intake/c2c/paper_chunks.jsonl", "text": "coverage preserving transfer avoids regression", "keywords": ["coverage", "transfer"]},
    ]
    rebuttal_chunks = [
        {"chunk_id": "rebuttal:collapse", "source_type": "rebuttal", "source_path": "intake/c2c/rebuttal_chunks.jsonl", "text": "reviewer warns about coverage collapse risk", "keywords": ["coverage", "collapse", "risk"]},
    ]
    code_chunks = [
        {"chunk_id": "code:rosetta/model/aligner.py", "source_type": "code", "path": "rosetta/model/aligner.py", "text": "aligner cache transfer implementation", "keywords": ["aligner", "cache"]},
        {"chunk_id": "code:rosetta/model/projector.py", "source_type": "code", "path": "rosetta/model/projector.py", "text": "projector cache routing implementation", "keywords": ["projector", "cache"]},
    ]
    chunk_index = {"entries": [*paper_chunks, *rebuttal_chunks, *code_chunks]}
    return chunk_index, paper_chunks, rebuttal_chunks, code_chunks


def test_evidence_request_plan_rejects_direction_fields() -> None:
    plan = default_c2c_evidence_request_plan(topic="cache")
    plan["direction_decision"] = {"direction_id": "bad"}

    errors = validate_c2c_evidence_request_plan(plan)

    assert "evidence_request_plan must not include direction_decision" in errors


def test_evidence_request_plan_requires_discriminating_uncertainty_contract() -> None:
    plan = default_c2c_evidence_request_plan(topic="cache")
    plan.pop("candidate_direction_hypotheses")
    plan["discriminating_evidence_requests"] = []

    errors = validate_c2c_evidence_request_plan(plan)

    assert "candidate_direction_hypotheses must contain at least two competing hypotheses" in errors
    assert "discriminating_evidence_requests must be a non-empty list" in errors


def test_deterministic_retriever_returns_stable_bundle_and_coverage() -> None:
    chunk_index, paper_chunks, rebuttal_chunks, code_chunks = _chunks()
    plan = default_c2c_evidence_request_plan(topic="cache")

    first_bundle, first_trace = retrieve_s1_c2c_requested_evidence(
        plan,
        chunk_index=chunk_index,
        paper_chunks=paper_chunks,
        rebuttal_chunks=rebuttal_chunks,
        code_chunks=code_chunks,
        implementation_surface_map={"surfaces": {"rosetta/model/aligner.py": {}, "rosetta/model/projector.py": {}}},
        negative_memory={"blocked_idea_patterns": ["hard gate collapse"]},
    )
    second_bundle, second_trace = retrieve_s1_c2c_requested_evidence(
        plan,
        chunk_index=chunk_index,
        paper_chunks=paper_chunks,
        rebuttal_chunks=rebuttal_chunks,
        code_chunks=code_chunks,
        implementation_surface_map={"surfaces": {"rosetta/model/aligner.py": {}, "rosetta/model/projector.py": {}}},
        negative_memory={"blocked_idea_patterns": ["hard gate collapse"]},
    )

    assert json.dumps(first_bundle, sort_keys=True) == json.dumps(second_bundle, sort_keys=True)
    assert first_trace["deterministic"] is True
    assert first_trace["coverage"]["paper"] >= 2
    assert first_trace["coverage"]["code"] >= 2
    assert first_trace["coverage"]["counterevidence"] >= 1
    assert first_bundle["producer"] == "deterministic_retriever"
    assert all(item["ref"] for item in first_bundle["items"])


def test_deterministic_retriever_prefers_allowed_code_surface_over_docs() -> None:
    plan = default_c2c_evidence_request_plan(topic="cache")
    plan["evidence_requests"] = [
        {
            "request_id": "code_surface",
            "source_type": "code",
            "query": "cache implementation surface aligner wrapper",
            "keywords": ["cache", "aligner", "wrapper"],
            "purpose": "implementation_surface",
            "top_k": 2,
            "filters": {},
            "must_resolve": True,
        }
    ]
    docs = [
        {
            "chunk_id": "README.md::README:0",
            "source_type": "code",
            "path": "README.md",
            "text": "cache implementation surface wrapper",
            "keywords": ["cache", "wrapper"],
        }
    ]
    code = [
        {
            "chunk_id": "code:rosetta/model/aligner.py",
            "source_type": "code",
            "path": "rosetta/model/aligner.py",
            "text": "aligner cache transfer implementation",
            "keywords": ["aligner", "cache"],
        },
        {
            "chunk_id": "code:rosetta/model/wrapper.py",
            "source_type": "code",
            "path": "rosetta/model/wrapper.py",
            "text": "wrapper cache routing implementation",
            "keywords": ["wrapper", "cache"],
        },
    ]

    bundle, trace = retrieve_s1_c2c_requested_evidence(
        plan,
        chunk_index={"entries": [*docs, *code]},
        code_chunks=[*docs, *code],
        implementation_surface_map={
            "surfaces": {
                "alignment_core": [{"path": "rosetta/model/aligner.py", "edit_surface": "allowed"}],
                "runtime_path": [{"path": "README.md", "edit_surface": "risky"}],
                "wrapper": [{"path": "rosetta/model/wrapper.py", "edit_surface": "allowed"}],
            }
        },
    )

    selected_paths = [item["source_path"] for item in bundle["items"]]
    assert selected_paths == ["rosetta/model/aligner.py", "rosetta/model/wrapper.py"]
    assert trace["coverage"]["code"] == 2


def test_must_resolve_rejects_same_source_zero_overlap_candidate() -> None:
    plan = default_c2c_evidence_request_plan(topic="cache")
    plan["evidence_requests"] = [
        {
            "request_id": "paper_router",
            "source_type": "paper",
            "query": "utility conditioned cache router",
            "keywords": ["utility", "router"],
            "purpose": "support",
            "top_k": 1,
            "filters": {},
            "must_resolve": True,
        }
    ]
    unrelated = {
        "chunk_id": "paper:unrelated",
        "source_type": "paper",
        "source_path": "intake/c2c/paper_chunks.jsonl",
        "text": "vision transformer image augmentation benchmark",
        "keywords": ["vision", "augmentation"],
    }

    bundle, trace = retrieve_s1_c2c_requested_evidence(
        plan,
        chunk_index={"entries": [unrelated]},
        paper_chunks=[unrelated],
    )

    assert bundle["items"] == []
    assert trace["unfilled_must_resolve_requests"][0]["request_id"] == "paper_router"
    assert any(item["reason"] == "source_only_match" for item in trace["rejected_top_candidates"])


def test_deterministic_retriever_expands_code_neighborhood_from_edges() -> None:
    plan = default_c2c_evidence_request_plan(topic="cache")
    plan["evidence_requests"] = [
        {
            "request_id": "code_surface",
            "source_type": "code",
            "query": "wrapper cache routing",
            "keywords": ["wrapper", "cache"],
            "purpose": "implementation_surface",
            "top_k": 1,
            "filters": {},
            "must_resolve": True,
        }
    ]
    code = [
        {
            "chunk_id": "wrapper_chunk",
            "source_type": "code",
            "path": "rosetta/model/wrapper.py",
            "text": "wrapper cache routing implementation",
            "keywords": ["wrapper", "cache"],
        },
        {
            "chunk_id": "router_chunk",
            "source_type": "code",
            "path": "rosetta/model/router.py",
            "text": "router applies utility control signal",
            "keywords": ["router", "utility"],
        },
    ]

    bundle, trace = retrieve_s1_c2c_requested_evidence(
        plan,
        chunk_index={"entries": code},
        code_chunks=code,
        code_edges=[{"edge_type": "resolved_call", "src": "wrapper_chunk", "dst": "router_chunk", "confidence": "test"}],
        implementation_surface_map={"surfaces": {"rosetta/model/wrapper.py": {"edit_surface": "allowed"}, "rosetta/model/router.py": {"edit_surface": "allowed"}}},
    )

    selected_paths = [item["source_path"] for item in bundle["items"]]
    assert selected_paths == ["rosetta/model/wrapper.py", "rosetta/model/router.py"]
    assert trace["code_neighborhood_expansions"][0]["reason"] == "callgraph_edge"
    assert "code_neighborhood" in trace["coverage_contributors"]


def test_deterministic_retriever_reports_unfilled_must_resolve() -> None:
    plan = default_c2c_evidence_request_plan(topic="cache")

    bundle, trace = retrieve_s1_c2c_requested_evidence(
        plan,
        chunk_index={"entries": []},
        paper_chunks=[],
        rebuttal_chunks=[],
        code_chunks=[],
    )

    assert bundle["items"] == []
    assert trace["unfilled_must_resolve_requests"]
    assert any(item["request_id"] == "paper_support" for item in trace["unfilled_must_resolve_requests"])
