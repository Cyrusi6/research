from __future__ import annotations

from pathlib import Path

import pytest

from auto_research.agents.experiment import _c2c_strict_evidence_inventory
from auto_research.research_state import IntegrityError, ResearchEventLedger
from support.authoritative_evidence import (
    record_completed_evidence_command,
    stage_authoritative_completion,
    validate_authoritative_completion,
)
from test_m113_ledger_closure import _running_attempt, _valid_completion
from test_m114_authoritative_phase_transactions import _c2c_inputs, _rehash_chain


def _passing_proxy_completion(project_root: Path) -> tuple[ResearchEventLedger, dict, dict]:
    attempt, trial_spec, comparison, baseline = _c2c_inputs(project_root)
    ledger = ResearchEventLedger(project_root)
    inventory = _c2c_strict_evidence_inventory(
        project_root=project_root,
        attempt=attempt,
        trial_spec=trial_spec,
        comparison_candidate=comparison,
        baseline=baseline,
        simulate=True,
    )
    completion = stage_authoritative_completion(
        project_root,
        attempt,
        trial_spec,
        inventory,
    )
    assert completion["schema_version"] == "auto_research_completion_evidence_v3"
    phase_execution = attempt["phase_executions"]["proxy"]
    assert completion["phase"] == "proxy"
    assert completion["phase_execution_id"] == phase_execution["phase_execution_id"]
    assert completion["producer_run_id"] == phase_execution["producer_run_id"]
    assert completion["command_plan_hash"] == phase_execution["command_plan_hash"]
    assert {entry["kind"] for entry in completion["entries"]} == {
        "activation_evidence",
        "full_s3_readiness",
        "proxy_baseline_fingerprint",
        "proxy_cache_report",
        "proxy_results",
    }
    assert all("outcome" not in entry and "observations" not in entry for entry in completion["entries"])
    return ledger, attempt, completion


def test_proxy_evidence_without_completed_command_receipt_lineage_is_zero_write_rejected(
    tmp_path: Path,
) -> None:
    baseline_root = tmp_path / "baseline"
    baseline_ledger, baseline_attempt, baseline_completion = _passing_proxy_completion(baseline_root)
    record_completed_evidence_command(
        baseline_root,
        baseline_ledger,
        baseline_attempt,
        baseline_completion,
    )
    validated = validate_authoritative_completion(
        baseline_root,
        baseline_ledger,
        baseline_attempt,
        baseline_completion,
    )
    assert validated.manifest["schema_version"] == "auto_research_evidence_manifest_v6"
    assert len(validated.lineage) == len(baseline_completion["entries"])
    assert all(item.receipt_hash and item.completed_event_id for item in validated.lineage.values())

    attack_root = tmp_path / "attack"
    attack_ledger, attack_attempt, forged_completion = _passing_proxy_completion(attack_root)
    before_state = attack_ledger.state()
    before_events = attack_ledger.events()

    with pytest.raises(IntegrityError, match="command|receipt|lineage"):
        attack_ledger.commit_proxy_evidence(forged_completion)

    assert attack_ledger.events() == before_events
    assert attack_ledger.state() == before_state
    assert before_state["attempts"][attack_attempt["attempt_id"]]["state"] == "PROXY_RUNNING"


def test_full_evidence_without_completed_command_receipt_lineage_is_zero_write_rejected(
    tmp_path: Path,
) -> None:
    baseline_root = tmp_path / "baseline"
    baseline_ledger, baseline_attempt = _running_attempt(baseline_root)
    baseline_completion = _valid_completion(baseline_root, baseline_attempt)
    record_completed_evidence_command(
        baseline_root,
        baseline_ledger,
        baseline_attempt,
        baseline_completion,
    )
    validated = validate_authoritative_completion(
        baseline_root,
        baseline_ledger,
        baseline_attempt,
        baseline_completion,
    )
    assert len(validated.observations) == 2
    assert {item["role"] for item in validated.observations} == {"baseline", "candidate"}

    attack_root = tmp_path / "attack"
    attack_ledger, attack_attempt = _running_attempt(attack_root)
    forged_completion = _valid_completion(attack_root, attack_attempt)
    before_state = attack_ledger.state()
    before_events = attack_ledger.events()

    with pytest.raises(IntegrityError, match="command|receipt|lineage"):
        attack_ledger.complete_attempt(forged_completion)

    assert attack_ledger.events() == before_events
    assert attack_ledger.state() == before_state
    assert before_state["attempts"][attack_attempt["attempt_id"]]["state"] == "FULL_RUNNING"


@pytest.mark.parametrize("phase", ["proxy", "full"])
def test_receipt_lineage_is_derived_only_after_completed_authoritative_output(
    tmp_path: Path,
    phase: str,
) -> None:
    if phase == "proxy":
        ledger, attempt, completion = _passing_proxy_completion(tmp_path)
    else:
        ledger, attempt = _running_attempt(tmp_path)
        completion = _valid_completion(tmp_path, attempt)

    with pytest.raises(ValueError, match="command|receipt|lineage"):
        validate_authoritative_completion(tmp_path, ledger, attempt, completion)

    record_completed_evidence_command(tmp_path, ledger, attempt, completion)
    validated = validate_authoritative_completion(tmp_path, ledger, attempt, completion)
    assert set(validated.lineage) == {entry["evidence_id"] for entry in completion["entries"]}
    for canonical_entry in validated.manifest["entries"]:
        assert canonical_entry["receipt_ref"]["digest"] == canonical_entry["receipt_hash"]
        assert canonical_entry["output_ref"]["digest"] == canonical_entry["content_hash"]
        assert canonical_entry["completed_event_id"]


def test_caller_supplied_receipt_lineage_must_equal_authoritative_derivation(tmp_path: Path) -> None:
    ledger, attempt = _running_attempt(tmp_path)
    completion = _valid_completion(tmp_path, attempt)
    record_completed_evidence_command(tmp_path, ledger, attempt, completion)
    validated = validate_authoritative_completion(tmp_path, ledger, attempt, completion)
    forged_manifest = validated.manifest
    forged_manifest["entries"][0]["command_hash"] = "0" * 64

    state = ledger.state()
    from auto_research.evidence_lineage import validate_receipt_bound_evidence

    with pytest.raises(ValueError, match="caller-supplied evidence lineage"):
        validate_receipt_bound_evidence(
            project_root=tmp_path,
            attempt=state["attempts"][attempt["attempt_id"]],
            trial_spec=attempt["frozen_trial_spec"],
            manifest=forged_manifest,
            phase_commands=state["phase_commands"],
            phase="full",
        )


def test_rebuild_rejects_rehashed_proxy_manifest_lineage_mutation(tmp_path: Path) -> None:
    ledger, attempt, completion = _passing_proxy_completion(tmp_path)
    record_completed_evidence_command(tmp_path, ledger, attempt, completion)
    committed_attempt, route = ledger.commit_proxy_evidence(completion)
    assert committed_attempt["state"] in {"PROXY_COMPLETED", "ABANDONED"}
    assert route["source"]["event_id"]
    assert ledger.rebuild()["last_sequence"] == ledger.state()["last_sequence"]

    event = next(item for item in ledger.events() if item["event_type"] == "ProxyEvidenceCommitted")
    attacked = dict(event["payload"])
    attacked["evidence_manifest"] = dict(attacked["evidence_manifest"])
    attacked["evidence_manifest"]["entries"] = [dict(item) for item in attacked["evidence_manifest"]["entries"]]
    attacked["evidence_manifest"]["entries"][0]["command_hash"] = "0" * 64
    attacked["proxy_outcome"] = dict(attacked["proxy_outcome"])
    from auto_research.domain_contracts import canonical_hash

    attacked["proxy_outcome"]["evidence_manifest_hash"] = canonical_hash(attacked["evidence_manifest"])
    _rehash_chain(ledger, event["sequence"], attacked)

    with pytest.raises(IntegrityError, match="canonical receipt-derived lineage|receipt-bound evidence"):
        ledger.rebuild()


def test_transaction_cache_revalidates_immutable_evidence_bytes(tmp_path: Path) -> None:
    ledger, attempt, completion = _passing_proxy_completion(tmp_path)
    record_completed_evidence_command(tmp_path, ledger, attempt, completion)
    ledger.commit_proxy_evidence(completion)
    events = ledger.events()
    event = next(item for item in events if item["event_type"] == "ProxyEvidenceCommitted")
    command_id = next(
        item["payload"]["command"]["command_id"]
        for item in events
        if item["event_type"] == "PhaseCommandStarted"
    )
    entry = event["payload"]["evidence_manifest"]["entries"][0]
    evidence_path = tmp_path / entry["relative_path"]
    evidence_path.write_text("{}", encoding="utf-8")

    with pytest.raises(IntegrityError, match="immutable receipt-bound evidence audit failed"):
        ledger.phase_command(command_id)
