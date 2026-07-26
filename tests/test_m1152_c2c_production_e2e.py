from __future__ import annotations

from pathlib import Path

import pytest

from auto_research.agents.experiment import ExperimentAgent
from auto_research.contract_store import ContractStore
from auto_research.research_state import ResearchEventLedger
from auto_research.validators import run_stage_gate
from support.local_c2c_execution import (
    build_c2c_context,
    create_local_c2c_repo,
    install_fake_gpu,
    invocation_records,
)


def _completed_receipts(root: Path) -> list[tuple[dict, dict, str]]:
    state = ResearchEventLedger(root).state()
    store = ContractStore(root)
    completed = []
    for record in state["phase_commands"].values():
        if record["status"] != "completed":
            continue
        receipt = store.read_json(record["receipt_ref"], schema_file="phase_run_receipt_v5.schema.json")
        completed.append((record["command"], receipt, record["receipt_ref"]["digest"]))
    return completed


def _assert_receipt_and_derivation_chain(root: Path) -> None:
    completed = _completed_receipts(root)
    store = ContractStore(root)
    assert completed
    assert all(receipt["exit_code"] == 0 for _, receipt, _ in completed)
    assert all(receipt["stdout_ref"]["digest"] == receipt["stdout_hash"] for _, receipt, _ in completed)
    assert all(receipt["stderr_ref"]["digest"] == receipt["stderr_hash"] for _, receipt, _ in completed)
    derivations = [
        (command, receipt, receipt_hash)
        for command, receipt, receipt_hash in completed
        if command["command_spec_id"].endswith("derive-evidence")
    ]
    assert derivations
    assert all(receipt["outputs"] for _, receipt, _ in derivations)
    physical_receipt_hashes = {
        receipt_hash
        for command, _, receipt_hash in completed
        if not command["command_spec_id"].endswith("derive-evidence")
    }
    derivation_by_receipt = {}
    for _, receipt, receipt_hash in derivations:
        assert receipt["derivation_ref"]
        assert receipt["derivation_hash"] == receipt["derivation_ref"]["digest"]
        derivation = store.read_json(
            receipt["derivation_ref"],
            schema_file="evidence_derivation_manifest_v2.schema.json",
        )
        assert {source["receipt_hash"] for source in derivation["source_commands"]}.issubset(
            physical_receipt_hashes
        )
        derivation_by_receipt[receipt_hash] = derivation
    state = ResearchEventLedger(root).state()
    for trial_result in state["trial_results"].values():
        evidence_manifest = trial_result["evidence_manifest"]
        entries = evidence_manifest["entries"]
        assert entries
        assert evidence_manifest["derive_receipt_hash"] in derivation_by_receipt
        assert evidence_manifest["derivation_ref"]["digest"] == evidence_manifest["derivation_hash"]
        for entry in entries:
            assert entry["receipt_ref"]
            assert entry["derivation_ref"]
            derivation = store.read_json(
                entry["derivation_ref"], schema_file="evidence_derivation_manifest_v2.schema.json"
            )
            normalized = next(
                output for output in derivation["normalized_outputs"] if output["kind"] == entry["kind"]
            )
            assert normalized["contract_ref"]["digest"] == entry["content_hash"]
            assert entry["receipt_hash"] == evidence_manifest["derive_receipt_hash"]
            assert entry["derivation_hash"] == evidence_manifest["derivation_hash"]
            assert derivation["source_commands"]


def _run_local_c2c(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, proxy_accuracy: float, profile: str):
    install_fake_gpu(tmp_path, monkeypatch)
    repo = create_local_c2c_repo(tmp_path, proxy_accuracy=proxy_accuracy)
    root = tmp_path / f"project-{profile}-{proxy_accuracy}"
    root.mkdir()
    context = build_c2c_context(root, repo, profile=profile)
    result = ExperimentAgent(context).run()
    return root, context, result


def test_c2c_real_proxy_reject_never_runs_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, context, result = _run_local_c2c(tmp_path, monkeypatch, proxy_accuracy=0.49, profile="standard")
    ledger = ResearchEventLedger(root)
    events = ledger.events()
    state = ledger.state()
    assert result.get("route_outcome", {}).get("next_action") == "PROPOSE_NEXT_VARIANT", result
    assert "ProxyEvidenceCommitted" in [event["event_type"] for event in events]
    assert "FullPhaseStarted" not in [event["event_type"] for event in events]
    records = invocation_records(root)
    assert records
    assert not any(record["kind"] == "train" and "/proxy/" not in record["config"] and "proxy_baseline" not in record["config"] for record in records)
    assert next(iter(state["directions"].values()))["budget"] == {"target": 5, "reserved": 0, "consumed": 0}
    _assert_receipt_and_derivation_chain(root)
    assert run_stage_gate("S3_experiment", root, context.config).to_dict()["status"] == "PASS"


def test_c2c_real_proxy_pass_commits_before_full_and_finalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, context, result = _run_local_c2c(tmp_path, monkeypatch, proxy_accuracy=0.51, profile="standard")
    ledger = ResearchEventLedger(root)
    events = ledger.events()
    event_types = [event["event_type"] for event in events]
    assert result.get("route_outcome", {}).get("next_action") == "PROPOSE_NEXT_VARIANT", result
    assert event_types.index("ProxyEvidenceCommitted") < event_types.index("FullPhaseStarted")
    assert event_types.index("FullPhaseStarted") < event_types.index("AttemptFinalized")
    records = invocation_records(root)
    assert any(record["kind"] == "train" and "/proxy/" not in record["config"] and "proxy_baseline" not in record["config"] for record in records)
    assert any(record["kind"] == "eval" and "/proxy/" not in record["config"] and "proxy_baseline" not in record["config"] for record in records)
    state = ledger.state()
    assert next(iter(state["directions"].values()))["budget"] == {"target": 5, "reserved": 0, "consumed": 1}
    assert len(state["trial_results"]) == 1
    _assert_receipt_and_derivation_chain(root)
    assert run_stage_gate("S3_experiment", root, context.config).to_dict()["status"] == "PASS"


def test_c2c_real_bootstrap_is_proxy_only_and_budget_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, context, result = _run_local_c2c(tmp_path, monkeypatch, proxy_accuracy=0.51, profile="bootstrap")
    ledger = ResearchEventLedger(root)
    state = ledger.state()
    event_types = [event["event_type"] for event in ledger.events()]
    assert result.get("route_outcome", {}).get("next_action") == "FINISH_RUN", result
    assert "FullPhaseStarted" not in event_types
    records = invocation_records(root)
    assert records
    assert not any(record["kind"] == "train" and "/proxy/" not in record["config"] and "proxy_baseline" not in record["config"] for record in records)
    assert next(iter(state["directions"].values()))["budget"] == {"target": 5, "reserved": 0, "consumed": 0}
    assert state["method_tried_history"] == []
    assert state["latest_direction_aggregate"] is None
    assert len(state["trial_results"]) == 1
    _assert_receipt_and_derivation_chain(root)
    assert run_stage_gate("S3_experiment", root, context.config).to_dict()["status"] == "PASS"
