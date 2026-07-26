from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from auto_research.agents.experiment import ExperimentAgent
from auto_research.derivation_contracts import build_readiness_check_plan, freeze_decoder_descriptor
from auto_research.derivation_validation import ReceiptBoundSources
from auto_research.proxy_classifier import derive_readiness_from_receipts
from auto_research.research_state import ResearchEventLedger
from support.local_c2c_execution import build_c2c_context, create_local_c2c_repo, install_fake_gpu


def test_c2c_command_plan_uses_core_derivation_not_virtual_collect(tmp_path: Path) -> None:
    agent = ExperimentAgent.__new__(ExperimentAgent)
    agent.context = SimpleNamespace(project_root=tmp_path)

    values = agent._c2c_command_spec_values(
        [("proxy_command_1", {"argv": ["python", "producer.py"]})],
        phase="proxy",
        cwd=tmp_path,
    )

    assert [item["command_spec_id"] for item in values] == [
        "proxy-proxy_command_1",
        "proxy-derive-evidence",
    ]
    assert values[-1]["dependencies"] == ["proxy-proxy_command_1"]
    source = inspect.getsource(ExperimentAgent._run_authoritative_step)
    assert "internal_output_command" not in source
    assert "collect-evidence" not in inspect.getsource(ExperimentAgent._execute_c2c_adapter_phase)


def test_receipt_authorized_readiness_block_is_not_overridden_by_activation_exit_zero(
    tmp_path: Path,
) -> None:
    decoder = freeze_decoder_descriptor(
        tmp_path,
        decoder_id="canonical-identity",
        decoder_version="1",
        semantic_contract={
            "canonicalization": {
                "encoding": "utf-8",
                "object_key_order": "lexicographic",
                "row_order": ["phase", "role", "dataset_id", "metric_id", "seed"],
                "duplicate_policy": "reject",
                "numeric_policy": "finite_non_boolean",
            },
            "coverage_contract": {
                "mode": "exact_cartesian",
                "datasets": ["readiness-dataset"],
                "seeds": [0],
                "metrics": ["readiness"],
                "roles": ["readiness"],
            },
        },
        authority_role_contract={"source_bindings": [{
            "source_ordinal": 0,
            "source_phase": "proxy",
            "command_spec_id": "proxy-readiness-command",
            "output_id": "raw-full-readiness",
            "authority_roles": ["readiness"],
            "readiness_check_ids": ["full-ready"],
        }]},
        output_contract={"expected_normalized_outputs": [{
            "ordinal": 0,
            "output_id": "normalized-full-readiness",
            "kind": "full_s3_readiness",
            "schema_version": "auto_research_full_s3_readiness_v4",
        }]},
    )
    binding = {
        "source_ordinal": 0,
        "source_phase": "proxy",
        "command_spec_id": "proxy-readiness-command",
        "output_id": "raw-full-readiness",
        "output_kind": "raw_readiness_check",
        "output_schema_version": "auto_research_raw_readiness_check_v1",
        "required_authority_roles": ["readiness"],
        "check_id": "full-ready",
    }
    plan = build_readiness_check_plan(
        plan_id="receipt-readiness-block",
        phase="proxy",
        checks=[{
            "ordinal": 0,
            "check_id": "full-ready",
            "check_kind": "raw_measurement",
            "source_bindings": [binding],
            "predicate": {"field_path": "ready", "comparator": "eq", "threshold": True},
            "required_coverage": {"mode": "exact", "expected_surface_ids": []},
            "decoder_descriptor": decoder,
            "blocked_classification": "IMPLEMENTATION_BLOCKED",
            "blocked_route": "REPAIR_IMPLEMENTATION",
        }],
    )
    key = ("proxy", "proxy-readiness-command", "raw-full-readiness")
    sources = ReceiptBoundSources(
        raw_facts={key: {"check_id": "full-ready", "ready": False, "activation_exit_code": 0}},
        raw_fact_lineage={key: {
            "source_phase": "proxy",
            "command_spec_id": "proxy-readiness-command",
            "output_id": "raw-full-readiness",
            "output_kind": "raw_readiness_check",
            "authority_roles": ["readiness"],
            "readiness_check_ids": ["full-ready"],
            "command_status": "completed",
            "exit_code": 0,
            "receipt_hash": "1" * 64,
            "receipt_ref": {"digest": "1" * 64},
            "output_ref": {"digest": "2" * 64},
            "completed_event_id": "event:readiness:completed",
        }},
        surface_checks=(),
        physical_inputs=(),
    )
    readiness = derive_readiness_from_receipts(
        readiness_check_plan=plan,
        receipt_bound_sources=sources,
    )
    assert readiness["ready"] is False
    assert [item["status"] for item in readiness["checks"]] == ["BLOCKED"]


def test_non_simulated_c2c_phase_commits_core_derivation_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    install_fake_gpu(tmp_path, monkeypatch)
    repo = create_local_c2c_repo(tmp_path / "fixture", proxy_accuracy=0.51)
    root = tmp_path / "c2c-derivation"
    root.mkdir()
    agent = ExperimentAgent(build_c2c_context(root, repo, profile="standard"))

    result = agent.run()

    assert result.get("attempt", {}).get("state") == "METHOD_COMPLETED", result
    events = ResearchEventLedger(root).events()
    command_events = [
        event
        for event in events
        if event["event_type"] == "PhaseCommandStarted"
    ]
    specs = [event["payload"]["command"]["command_spec_id"] for event in command_events]
    assert "proxy-derive-evidence" in specs
    assert "full-derive-evidence" in specs
    assert not any("collect-evidence" in spec for spec in specs)
    phase_commands = ResearchEventLedger(root).state()["phase_commands"]
    derivations = [
        record
        for record in phase_commands.values()
        if record["command"]["command_spec_id"].endswith("-derive-evidence")
    ]
    assert len(derivations) == 2
    assert all(record["status"] == "completed" and record["receipt_ref"] for record in derivations)
