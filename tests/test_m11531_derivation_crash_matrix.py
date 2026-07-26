from __future__ import annotations

from pathlib import Path

import pytest

from auto_research.research_state import ResearchEventLedger
from support.m11531_crash_harness import (
    CRASH_AFTER_DERIVED_BLOBS_BEFORE_LOCATOR,
    CRASH_AFTER_DERIVE_COMPLETED_BEFORE_EVIDENCE,
    CRASH_AFTER_DERIVE_RECEIPT_BEFORE_COMPLETED,
    CRASH_AFTER_PHYSICAL_BEFORE_DERIVE_STARTED,
    CRASH_AFTER_PROXY_COMMIT_BEFORE_ROUTE,
    CRASH_AFTER_READINESS_COMPLETED_BEFORE_PROXY_COMMIT,
    CRASH_AFTER_READINESS_RECEIPT_BEFORE_COMPLETED,
    CrashObservation,
    assert_event_chain_prefix,
    assert_injected_crash,
    assert_legal_baseline,
    assert_physical_not_replayed,
    create_crash_project,
    observe_authority,
    raw_event_rows,
    run_cold_agent,
)


@pytest.fixture(scope="module")
def legal_non_crash_baseline(
    tmp_path_factory: pytest.TempPathFactory,
) -> CrashObservation:
    base = tmp_path_factory.mktemp("m11531-crash-control")
    monkeypatch = pytest.MonkeyPatch()
    try:
        root, repo = create_crash_project(
            base,
            monkeypatch,
            name="legal-proxy-reject-control",
            proxy_accuracy=0.49,
        )
        completed = run_cold_agent(root, repo)
        observation = observe_authority(root)
        assert_legal_baseline(observation, completed)
        return observation
    finally:
        monkeypatch.undo()


def test_crash_after_all_physical_receipts_before_derive_started_recovers_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legal_non_crash_baseline: CrashObservation,
) -> None:
    assert legal_non_crash_baseline.proxy_evidence_committed == 1
    root, repo = create_crash_project(
        tmp_path,
        monkeypatch,
        name="physical-before-derive-started",
        proxy_accuracy=0.49,
    )

    crashed = run_cold_agent(
        root,
        repo,
        crash_point=CRASH_AFTER_PHYSICAL_BEFORE_DERIVE_STARTED,
    )
    assert_injected_crash(
        crashed,
        "after physical receipts before derive PhaseCommandStarted",
    )
    before_rows = raw_event_rows(root)
    before = observe_authority(root)
    assert before.phase_command_started == before.phase_command_completed, before.diagnostic()
    assert before.proxy_evidence_committed == 0, before.diagnostic()
    assert before.full_phase_started == 0, before.diagnostic()
    assert before.physical_invocations > 0, before.diagnostic()
    assert before.physical_max_repeat == 1, before.diagnostic()

    restarted = run_cold_agent(root, repo)
    after = observe_authority(root)

    assert restarted.returncode == 0, restarted.stderr
    assert restarted.result is not None and restarted.result["result_route"] == "PROPOSE_NEXT_VARIANT"
    assert_event_chain_prefix(before_rows, raw_event_rows(root))
    assert_physical_not_replayed(before, after)
    assert after.producing_derive_by_phase == {"proxy": 1}, after.diagnostic()
    assert after.validator_recomputations > before.validator_recomputations, after.diagnostic()
    assert after.derive_receipt_hash is not None, after.diagnostic()
    assert after.derivation_manifest_hash is not None, after.diagnostic()
    assert after.proxy_decisions == ("PROPOSE_NEXT_VARIANT",), after.diagnostic()
    assert after.proxy_evidence_committed == 1, after.diagnostic()
    assert after.attempt_finalized == 0, after.diagnostic()
    assert after.full_phase_started == 0, after.diagnostic()
    assert after.budget == {"target": 5, "reserved": 0, "consumed": 0}, after.diagnostic()


def test_crash_after_normalized_manifest_before_durable_derive_receipt_blocks_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legal_non_crash_baseline: CrashObservation,
) -> None:
    assert legal_non_crash_baseline.derivation_manifest_hash is not None
    root, repo = create_crash_project(
        tmp_path,
        monkeypatch,
        name="derived-blobs-before-locator",
        proxy_accuracy=0.49,
    )

    crashed = run_cold_agent(
        root,
        repo,
        crash_point=CRASH_AFTER_DERIVED_BLOBS_BEFORE_LOCATOR,
    )
    assert_injected_crash(
        crashed,
        "after normalized blobs and derivation manifest before locator",
    )
    before_rows = raw_event_rows(root)
    before = observe_authority(root)
    assert before.phase_command_started == before.phase_command_completed + 1, before.diagnostic()
    assert before.derive_receipt_hash is None, before.diagnostic()
    assert before.derivation_manifest_hash is None, before.diagnostic()
    assert before.orphan_derivation_manifest_hashes, before.diagnostic()
    orphan_hashes = before.orphan_derivation_manifest_hashes

    restarted = run_cold_agent(root, repo)
    after = observe_authority(root)

    assert restarted.returncode == 0, restarted.stderr
    assert restarted.result is not None and restarted.result["result_route"] == "BLOCK_INTEGRITY"
    assert_event_chain_prefix(before_rows, raw_event_rows(root))
    assert_physical_not_replayed(before, after)
    assert after.producing_derive_by_phase == {"proxy": 1}, after.diagnostic()
    assert after.phase_command_unknown == 1, after.diagnostic()
    assert after.last_route == "BLOCK_INTEGRITY", after.diagnostic()
    assert after.orphan_derivation_manifest_hashes == orphan_hashes, after.diagnostic()
    assert after.proxy_evidence_committed == 0, after.diagnostic()
    assert after.attempt_finalized == 0, after.diagnostic()
    assert after.full_phase_started == 0, after.diagnostic()
    assert after.budget == {"target": 5, "reserved": 0, "consumed": 0}, after.diagnostic()


def test_crash_after_durable_derive_receipt_before_completed_reconciles_without_rederive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legal_non_crash_baseline: CrashObservation,
) -> None:
    assert legal_non_crash_baseline.producing_derive_by_phase == {"proxy": 1}
    root, repo = create_crash_project(
        tmp_path,
        monkeypatch,
        name="derive-receipt-before-completed",
        proxy_accuracy=0.49,
    )

    crashed = run_cold_agent(
        root,
        repo,
        crash_point=CRASH_AFTER_DERIVE_RECEIPT_BEFORE_COMPLETED,
    )
    assert_injected_crash(
        crashed,
        "after durable derive receipt before PhaseCommandCompleted",
    )
    before_rows = raw_event_rows(root)
    before = observe_authority(root)
    assert before.phase_command_started == before.phase_command_completed + 1, before.diagnostic()
    assert before.derive_receipt_hash is not None, before.diagnostic()
    assert before.derivation_manifest_hash is not None, before.diagnostic()
    frozen_receipt_hash = before.derive_receipt_hash
    frozen_manifest_hash = before.derivation_manifest_hash

    restarted = run_cold_agent(root, repo)
    after = observe_authority(root)

    assert restarted.returncode == 0, restarted.stderr
    assert restarted.result is not None and restarted.result["result_route"] == "PROPOSE_NEXT_VARIANT"
    assert_event_chain_prefix(before_rows, raw_event_rows(root))
    assert_physical_not_replayed(before, after)
    assert after.producing_derive_by_phase == {"proxy": 1}, after.diagnostic()
    assert after.validator_recomputations > before.validator_recomputations, after.diagnostic()
    assert after.derive_receipt_hash == frozen_receipt_hash, after.diagnostic()
    assert after.derivation_manifest_hash == frozen_manifest_hash, after.diagnostic()
    assert after.phase_command_unknown == 0, after.diagnostic()
    assert after.proxy_evidence_committed == 1, after.diagnostic()
    assert after.full_phase_started == 0, after.diagnostic()
    assert after.budget == {"target": 5, "reserved": 0, "consumed": 0}, after.diagnostic()


def test_crash_after_derive_completed_before_proxy_evidence_commit_reuses_manifest_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legal_non_crash_baseline: CrashObservation,
) -> None:
    assert legal_non_crash_baseline.validator_recomputations > 0
    root, repo = create_crash_project(
        tmp_path,
        monkeypatch,
        name="derive-completed-before-evidence",
        proxy_accuracy=0.49,
    )

    crashed = run_cold_agent(
        root,
        repo,
        crash_point=CRASH_AFTER_DERIVE_COMPLETED_BEFORE_EVIDENCE,
    )
    assert_injected_crash(
        crashed,
        "after derive Completed before ProxyEvidenceCommitted",
    )
    before_rows = raw_event_rows(root)
    before = observe_authority(root)
    assert before.phase_command_started == before.phase_command_completed, before.diagnostic()
    assert before.proxy_evidence_committed == 0, before.diagnostic()
    assert before.derive_receipt_hash is not None, before.diagnostic()
    assert before.derivation_manifest_hash is not None, before.diagnostic()

    restarted = run_cold_agent(root, repo)
    after = observe_authority(root)

    assert restarted.returncode == 0, restarted.stderr
    assert restarted.result is not None and restarted.result["result_route"] == "PROPOSE_NEXT_VARIANT"
    assert_event_chain_prefix(before_rows, raw_event_rows(root))
    assert_physical_not_replayed(before, after)
    assert after.producing_derive_by_phase == {"proxy": 1}, after.diagnostic()
    assert after.derive_receipt_hash == before.derive_receipt_hash, after.diagnostic()
    assert after.derivation_manifest_hash == before.derivation_manifest_hash, after.diagnostic()
    assert after.proxy_evidence_committed == 1, after.diagnostic()
    assert after.full_phase_started == 0, after.diagnostic()
    assert after.attempt_finalized == 0, after.diagnostic()
    assert after.budget == {"target": 5, "reserved": 0, "consumed": 0}, after.diagnostic()


def test_crash_after_readiness_receipt_before_completed_reconciles_physical_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legal_non_crash_baseline: CrashObservation,
) -> None:
    assert legal_non_crash_baseline.physical_max_repeat == 1
    root, repo = create_crash_project(
        tmp_path,
        monkeypatch,
        name="readiness-receipt-before-completed",
        proxy_accuracy=0.49,
    )

    crashed = run_cold_agent(
        root,
        repo,
        crash_point=CRASH_AFTER_READINESS_RECEIPT_BEFORE_COMPLETED,
    )
    assert_injected_crash(
        crashed,
        "after readiness physical receipt before PhaseCommandCompleted",
    )
    before_rows = raw_event_rows(root)
    before = observe_authority(root)
    readiness_records = _readiness_command_records(root)
    assert len(readiness_records) == 1 and readiness_records[0]["status"] == "started"
    assert before.phase_command_started == before.phase_command_completed + 1, before.diagnostic()
    assert before.producing_derive_invocations == 0, before.diagnostic()
    assert before.physical_invocations > 0, before.diagnostic()

    restarted = run_cold_agent(root, repo)
    after = observe_authority(root)

    assert restarted.returncode == 0, restarted.stderr
    assert restarted.result is not None and restarted.result["result_route"] == "PROPOSE_NEXT_VARIANT"
    assert_event_chain_prefix(before_rows, raw_event_rows(root))
    assert_physical_not_replayed(before, after)
    readiness_records = _readiness_command_records(root)
    assert len(readiness_records) == 1 and readiness_records[0]["status"] == "completed"
    assert after.producing_derive_by_phase == {"proxy": 1}, after.diagnostic()
    assert after.validator_recomputations > before.validator_recomputations, after.diagnostic()
    assert after.proxy_decisions == ("PROPOSE_NEXT_VARIANT",), after.diagnostic()
    assert after.proxy_evidence_committed == 1, after.diagnostic()
    assert after.full_phase_started == 0, after.diagnostic()
    assert after.budget == {"target": 5, "reserved": 0, "consumed": 0}, after.diagnostic()


def test_crash_after_readiness_completed_before_proxy_commit_preserves_run_full_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legal_non_crash_baseline: CrashObservation,
) -> None:
    assert legal_non_crash_baseline.full_phase_started == 0
    root, repo = create_crash_project(
        tmp_path,
        monkeypatch,
        name="readiness-completed-before-proxy-commit",
        proxy_accuracy=0.51,
    )

    crashed = run_cold_agent(
        root,
        repo,
        crash_point=CRASH_AFTER_READINESS_COMPLETED_BEFORE_PROXY_COMMIT,
    )
    assert_injected_crash(
        crashed,
        "after readiness Completed before proxy evidence commit",
    )
    before_rows = raw_event_rows(root)
    before = observe_authority(root)
    readiness_records = _readiness_command_records(root)
    assert len(readiness_records) == 1 and readiness_records[0]["status"] == "completed"
    assert before.producing_derive_invocations == 0, before.diagnostic()
    assert before.proxy_evidence_committed == 0, before.diagnostic()
    assert before.full_phase_started == 0, before.diagnostic()

    restarted = run_cold_agent(root, repo)
    after = observe_authority(root)

    assert restarted.returncode == 0, restarted.stderr
    assert restarted.result is not None and restarted.result["result_route"] == "PROPOSE_NEXT_VARIANT"
    assert_event_chain_prefix(before_rows, raw_event_rows(root))
    assert after.physical_invocations > before.physical_invocations, after.diagnostic()
    assert after.physical_unique_invocations == after.physical_invocations, after.diagnostic()
    assert after.physical_max_repeat == 1, after.diagnostic()
    assert after.producing_derive_by_phase == {"proxy": 1, "full": 1}, after.diagnostic()
    assert after.validator_recomputations > before.validator_recomputations, after.diagnostic()
    assert after.proxy_decisions == ("RUN_FULL",), after.diagnostic()
    assert after.event_types.index("ProxyEvidenceCommitted") < after.event_types.index("FullPhaseStarted")
    assert after.full_phase_started == 1, after.diagnostic()
    assert after.proxy_evidence_committed == 1, after.diagnostic()
    assert after.attempt_finalized == 1, after.diagnostic()
    assert after.budget == {"target": 5, "reserved": 0, "consumed": 1}, after.diagnostic()


def test_crash_after_proxy_evidence_commit_before_route_delivery_replays_blocked_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legal_non_crash_baseline: CrashObservation,
) -> None:
    assert legal_non_crash_baseline.last_route == "PROPOSE_NEXT_VARIANT"
    root, repo = create_crash_project(
        tmp_path,
        monkeypatch,
        name="proxy-commit-before-route",
        proxy_accuracy=0.51,
        readiness_blocked=True,
    )

    crashed = run_cold_agent(
        root,
        repo,
        crash_point=CRASH_AFTER_PROXY_COMMIT_BEFORE_ROUTE,
    )
    assert_injected_crash(
        crashed,
        "after ProxyEvidenceCommitted before route delivery",
    )
    before_rows = raw_event_rows(root)
    before = observe_authority(root)
    assert before.last_route == "REPAIR_IMPLEMENTATION", before.diagnostic()
    assert before.proxy_decisions == ("REPAIR_IMPLEMENTATION",), before.diagnostic()
    assert before.proxy_evidence_committed == 1, before.diagnostic()
    assert before.full_phase_started == 0, before.diagnostic()
    assert before.attempt_finalized == 0, before.diagnostic()
    assert before.budget == {"target": 5, "reserved": 1, "consumed": 0}, before.diagnostic()

    restarted = run_cold_agent(root, repo)
    after = observe_authority(root)

    assert restarted.returncode == 0, restarted.stderr
    assert restarted.result is not None and restarted.result["result_route"] == "REPAIR_IMPLEMENTATION"
    assert restarted.result["result_source_event_id"] == before.last_route_source_event_id
    assert raw_event_rows(root) == before_rows
    assert_physical_not_replayed(before, after)
    assert after.producing_derive_by_phase == {"proxy": 1}, after.diagnostic()
    assert after.derive_receipt_hash == before.derive_receipt_hash, after.diagnostic()
    assert after.derivation_manifest_hash == before.derivation_manifest_hash, after.diagnostic()
    assert after.last_route_source_event_id == before.last_route_source_event_id, after.diagnostic()
    assert after.proxy_evidence_committed == 1, after.diagnostic()
    assert after.full_phase_started == 0, after.diagnostic()
    assert after.attempt_finalized == 0, after.diagnostic()
    assert after.budget == {"target": 5, "reserved": 1, "consumed": 0}, after.diagnostic()


def _readiness_command_records(root: Path) -> list[dict]:
    state = ResearchEventLedger(root).state()
    return [
        record
        for record in state["phase_commands"].values()
        if "activation_smoke" in str(record["command"]["command_spec_id"])
    ]
