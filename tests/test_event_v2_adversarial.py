from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from auto_research.domain_contracts import canonical_hash, implementation_hash
from auto_research.research_state import IntegrityError, ResearchEventLedger
from test_authoritative_state_machine import (
    _complete,
    _direction,
    _initialize,
    _reserve,
    _trial_spec,
    _variant,
)
from test_m113_ledger_closure import (
    _failure_evidence as _canonical_failure_evidence,
    _resume_evidence,
)
from support.authoritative_evidence import build_failure_evidence_v4, start_attempt_phase


def _start_execution(ledger: ResearchEventLedger, attempt: dict) -> dict:
    phase = "proxy" if attempt["attempt_kind"] in {"proxy", "bootstrap_proxy"} else "full"
    return start_attempt_phase(ledger, attempt, phase)


def _reserve_proxy(ledger: ResearchEventLedger, direction: dict, variant: dict) -> dict:
    _initialize(ledger, direction, variant)
    return ledger.reserve_attempt(
        profile="standard",
        direction=direction,
        variant=variant,
        implementation_hash=canonical_hash({"variant": variant["variant_spec_hash"]}),
        attempt_kind="proxy",
        trial_spec=_trial_spec(
            profile="standard",
            attempt_kind="proxy",
            project_root=ledger.project_root,
        ),
    )


def _repair(ledger: ResearchEventLedger, attempt: dict, revision: int) -> dict:
    return ledger.revise_implementation(
        attempt["attempt_id"],
        implementation_hash=implementation_hash(
            frozen_patch={"revision": revision},
            files={"src/router.py": f"revision-{revision}"},
            manifest={"revision": revision},
        ),
    )


def test_repair_revision_can_reenter_same_proxy_running_transition(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    attempt = _reserve_proxy(ledger, direction, variant)
    running = _start_execution(ledger, attempt)
    failed, _ = ledger.disposition_failure(
        build_failure_evidence_v4(
            tmp_path,
            running,
            failure_class="activation_failure",
            suffix="first",
        )
    )
    repaired = _repair(ledger, failed, 2)
    rerun = _start_execution(ledger, repaired)
    assert rerun["state"] == "PROXY_RUNNING"
    transitions = [event for event in ledger.events() if event["event_type"] == "ProxyPhaseStarted"]
    assert len(transitions) == 2
    assert transitions[0]["event_id"] != transitions[1]["event_id"]


def test_same_failure_class_in_new_revision_creates_new_disposition_event(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    attempt = _reserve_proxy(ledger, direction, variant)
    first_running = _start_execution(ledger, attempt)
    first_failed, _ = ledger.disposition_failure(
        build_failure_evidence_v4(
            tmp_path,
            first_running,
            failure_class="activation_failure",
            suffix="first",
        )
    )
    repaired = _repair(ledger, first_failed, 2)
    second_running = _start_execution(ledger, repaired)
    second_failed, _ = ledger.disposition_failure(
        build_failure_evidence_v4(
            tmp_path,
            second_running,
            failure_class="activation_failure",
            suffix="second",
        )
    )
    events = [event for event in ledger.events() if event["event_type"] == "AttemptDispositioned"]
    assert len(events) == 2
    assert events[0]["event_id"] != events[1]["event_id"]
    assert second_failed["lifecycle_generation"] == 1


def test_late_finalize_replay_returns_original_attempt_route(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    _initialize(ledger, direction, _variant(direction, 1))
    first = _reserve(ledger, direction, _variant(direction, 1))
    first_completed, first_route = _complete(ledger, first, outcome="rejected")
    second_variant = _variant(direction, 2)
    _initialize(ledger, direction, second_variant)
    second = _reserve(ledger, direction, second_variant)
    _complete(ledger, second, outcome="rejected")
    historical = ledger.query_operation_result(first_route["source"]["event_id"])
    replayed_attempt = historical["attempt"]
    replayed_route = historical["route_outcome"]
    assert replayed_attempt == first_completed
    assert replayed_route == first_route
    assert replayed_route != ledger.state()["last_route_outcome"]


def test_late_disposition_replay_returns_original_attempt_route(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    _initialize(ledger, direction, _variant(direction, 1))
    first = _reserve(ledger, direction, _variant(direction, 1))
    running = _start_execution(ledger, first)
    evidence = _canonical_failure_evidence(
        tmp_path,
        running,
        failure_class="resource_pause",
        exit_code=137,
        resource_type="system_memory",
    )
    historical_attempt, historical_route = ledger.disposition_failure(evidence, event_id="pause:first")
    ledger.resume_attempt(_resume_evidence(tmp_path, ledger, historical_attempt, resource_type="system_memory"))
    before = len(ledger.events())
    replayed_attempt, replayed_route = ledger.disposition_failure(evidence, event_id="pause:first")
    assert len(ledger.events()) == before
    assert replayed_attempt == historical_attempt
    assert replayed_route == historical_route


@pytest.mark.parametrize("mutation", ["next_action", "source_event_id", "budget_snapshot", "idempotency_key"])
def test_low_level_append_rejects_forged_finalization_route(tmp_path: Path, mutation: str) -> None:
    ledger = ResearchEventLedger(tmp_path)
    before = ledger.events()
    with pytest.raises(IntegrityError, match="only permits AuditMarker"):
        ledger.append("AttemptFinalized", {"mutation": mutation}, event_id=f"forged-final:{mutation}")
    assert ledger.events() == before


def test_low_level_append_rejects_failure_target_and_action_mismatch(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    before = ledger.events()
    with pytest.raises(IntegrityError, match="only permits AuditMarker"):
        ledger.append("AttemptDispositioned", {"failure_class": "activation_failure", "target_state": "RESOURCE_PAUSED", "next_action": "FINISH_RUN"}, event_id="forged-disposition:mismatch")
    assert ledger.events() == before


@pytest.mark.parametrize("mutation", ["next_action", "source_event_id", "budget_snapshot", "idempotency_key"])
def test_low_level_append_rejects_forged_disposition_route(tmp_path: Path, mutation: str) -> None:
    ledger = ResearchEventLedger(tmp_path)
    before = ledger.events()
    with pytest.raises(IntegrityError, match="only permits AuditMarker"):
        ledger.append("AttemptDispositioned", {"mutation": mutation}, event_id=f"forged-disposition:{mutation}")
    assert ledger.events() == before


def test_stale_projection_writer_cannot_overwrite_newer_sequence(tmp_path: Path) -> None:
    first = ResearchEventLedger(tmp_path)
    second = ResearchEventLedger(tmp_path)
    first_committed = threading.Event()
    allow_stale_write = threading.Event()
    original_write = first._write_projections

    def delayed_write(state: dict) -> None:
        first_committed.set()
        assert allow_stale_write.wait(timeout=5)
        original_write(state)

    first._write_projections = delayed_write
    thread = threading.Thread(target=lambda: first.append("AuditMarker", {"index": 1}, event_id="audit:one"))
    thread.start()
    assert first_committed.wait(timeout=5)
    second.append("AuditMarker", {"index": 2}, event_id="audit:two")
    allow_stale_write.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    snapshot = json.loads(first.snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["last_sequence"] == 2
    assert len(first.events()) == 2


def test_three_repair_generations_keep_one_attempt_and_consume_once(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    for revision in (2, 3):
        running = _start_execution(ledger, ledger.state()["attempts"][attempt["attempt_id"]])
        failed, _ = ledger.disposition_failure(
            build_failure_evidence_v4(
                tmp_path,
                running,
                failure_class="activation_failure",
                suffix=str(revision),
            )
        )
        _repair(ledger, failed, revision)
    current = ledger.state()["attempts"][attempt["attempt_id"]]
    completed, _ = _complete(ledger, current, outcome="accepted")
    state = ledger.state()
    assert completed["attempt_id"] == attempt["attempt_id"]
    assert completed["variant_spec_hash"] == attempt["variant_spec_hash"]
    assert completed["lifecycle_generation"] == 2
    assert len(completed["implementation_revisions"]) == 3
    assert state["directions"][direction["direction_semantic_hash"]]["budget"] == {"target": 5, "reserved": 0, "consumed": 1}
    assert len(state["method_tried_history"]) == 1


def test_concurrent_plan_and_reserve_enforces_single_execution_slot(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    ledger.select_direction(direction)

    def compete(index: int) -> str | None:
        variant = _variant(direction, index)
        local = ResearchEventLedger(tmp_path)
        try:
            local.plan_variant(variant)
            spec = _trial_spec(
                profile="standard",
                attempt_kind="full",
                project_root=tmp_path,
            )
            return local.reserve_attempt(
                profile="standard",
                direction=direction,
                variant=variant,
                implementation_hash=canonical_hash({"variant": index}),
                attempt_kind="full",
                trial_spec=spec,
            )["attempt_id"]
        except IntegrityError:
            return None

    with ThreadPoolExecutor(max_workers=16) as executor:
        winners = [attempt_id for attempt_id in executor.map(compete, range(1, 17)) if attempt_id]
    assert len(winners) == 1
    events = ledger.events()
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    state = ledger.rebuild()
    assert sum(1 for item in state["attempts"].values() if item["reserved_slot"]) == 1
