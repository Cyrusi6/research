from auto_research.agents.literature import _normalize_s1_c2c_direction_payload, _validate_s1_c2c_direction_payload


def test_s1_c2c_direction_payload_derives_expected_files_from_bundle_code_refs() -> None:
    paper_ref = {"source_type": "paper", "source_label": "paper:1", "chunk_id": "paper:1", "claim": "Paper support."}
    code_ref = {
        "source_type": "code",
        "source_label": "wrapper:forward",
        "chunk_id": "wrapper:forward",
        "source_path": "/home/user/projects/C2C/rosetta/model/wrapper.py",
        "claim": "Wrapper editable surface.",
    }
    counter_ref = {"source_type": "rebuttal", "source_label": "rebuttal:1", "chunk_id": "rebuttal:1", "claim": "Reviewer risk."}
    evidence_bundle = {
        "schema_version": "c2c_s1_deterministic_evidence_bundle_v1",
        "items": [
            {"evidence_id": "ev_paper", "source_type": "paper", "ref": paper_ref, **paper_ref},
            {"evidence_id": "ev_code", "source_type": "code", "ref": code_ref, **code_ref},
            {"evidence_id": "ev_counter", "source_type": "rebuttal", "ref": counter_ref, **counter_ref, "risks": ["counterevidence"]},
        ],
    }
    payload = {
        "schema_version": "c2c_s1_direction_agent_v1",
        "status": "ok",
        "direction_decision": {
            "direction_id": "utility_predicted_cache_routing",
            "mechanism_direction": "Utility Predicted Cache Routing",
            "mechanism_type": "utility_predicted_cache_routing",
            "mechanism_axis": "routing",
            "integration_point": "wrapper",
            "control_signal": "utility",
            "core_hypothesis": "Use utility-aware routing.",
            "why_baseline_fails": "The baseline has no utility signal.",
            "why_this_direction": "The bundle supports wrapper routing.",
            "required_evidence_refs": [paper_ref],
            "counterevidence_refs": [counter_ref],
            "implementation_surface_refs": ["rosetta/model/wrapper.py"],
        },
        "selected_ideas": [
            {
                "id": "utility_predicted_cache_routing",
                "title": "Utility Predicted Cache Routing",
                "hypothesis": "Use utility-aware routing.",
                "novelty_score": 0.7,
                "feasibility_score": 0.8,
                "verification_commands": ["py_compile"],
                "evidence_refs": [paper_ref],
                "counterevidence_refs": [counter_ref],
                "code_refs": ["rosetta/model/wrapper.py"],
                "reviewer_risk_response": "Probe reviewer risk.",
                "mechanism_type": "utility_predicted_cache_routing",
            }
        ],
        "negative_constraints": {},
    }

    normalized = _normalize_s1_c2c_direction_payload(payload, evidence_bundle=evidence_bundle)

    assert normalized["direction_decision"]["expected_files"] == ["rosetta/model/wrapper.py"]
    assert normalized["selected_ideas"][0]["expected_files"] == ["rosetta/model/wrapper.py"]
    assert normalized["direction_decision"]["implementation_surface_refs"] == [code_ref]
    assert normalized["selected_ideas"][0]["code_refs"] == [code_ref]
    assert _validate_s1_c2c_direction_payload(normalized, evidence_bundle=evidence_bundle) == []
