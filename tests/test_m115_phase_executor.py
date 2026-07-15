from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from auto_research.phase_execution import (
    AuthoritativePhaseContext,
    C2CFullPhaseExecutor,
    C2CProxyPhaseExecutor,
    GenericExternalPhaseExecutor,
    PhaseArtifactInventory,
    PhaseExecutor,
    SyntheticPhaseExecutor,
    TypedPhaseFailure,
)


def _context(tmp_path: Path, *, phase: str = "proxy") -> AuthoritativePhaseContext:
    authorization = None
    if phase == "full":
        authorization = {
            "event_id": "event:proxy-committed",
            "event_hash": "f" * 64,
            "outcome_hash": "e" * 64,
            "decision": "RUN_FULL",
        }
    return AuthoritativePhaseContext(
        project_root=tmp_path,
        attempt_id="attempt-0001",
        direction_semantic_hash="1" * 64,
        direction_spec_hash="2" * 64,
        variant_semantic_hash="3" * 64,
        variant_spec_hash="4" * 64,
        trial_spec_hash="5" * 64,
        lifecycle_generation=2,
        implementation_hash="6" * 64,
        attempt_input_hash="7" * 64,
        phase=phase,
        phase_execution_id=f"phase-{phase}-0001",
        phase_start_event_id=f"event:{phase}-started",
        producer_run_id=f"producer-{phase}-0001",
        proxy_authorization=authorization,
    )


def _inventory(context: AuthoritativePhaseContext, *, kind: str = "main_results") -> PhaseArtifactInventory:
    return PhaseArtifactInventory.from_artifacts(
        context,
        [
            {
                "kind": kind,
                "source_path": f"experiment/staging/{context.producer_run_id}/{kind}.json",
                "producer_run_id": context.producer_run_id,
            }
        ],
    )


def test_authority_checker_runs_before_phase_runner(tmp_path: Path) -> None:
    context = _context(tmp_path)
    calls: list[str] = []

    def checker(received: AuthoritativePhaseContext) -> bool:
        calls.append("authority")
        assert received == context
        return True

    def runner(received: AuthoritativePhaseContext) -> PhaseArtifactInventory:
        calls.append("runner")
        return _inventory(received, kind="proxy_results")

    result = C2CProxyPhaseExecutor(authority_checker=checker, runner=runner).execute(context)

    assert calls == ["authority", "runner"]
    assert result.context == context
    assert isinstance(C2CProxyPhaseExecutor(checker, runner), PhaseExecutor)


def test_rejected_authority_never_invokes_runner(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(context: AuthoritativePhaseContext) -> PhaseArtifactInventory:
        calls.append("runner")
        return _inventory(context)

    executor = GenericExternalPhaseExecutor(authority_checker=lambda context: False, runner=runner)

    with pytest.raises(TypedPhaseFailure) as raised:
        executor.execute(_context(tmp_path))

    assert raised.value.failure_class == "authority_rejected"
    assert raised.value.retryable is False
    assert calls == []


def test_authority_identity_drift_is_rejected_before_runner(tmp_path: Path) -> None:
    context = _context(tmp_path)
    authoritative = replace(context, lifecycle_generation=context.lifecycle_generation + 1)
    calls: list[str] = []
    executor = SyntheticPhaseExecutor(
        authority_checker=lambda received: authoritative,
        runner=lambda received: calls.append("runner") or _inventory(received),
    )

    with pytest.raises(TypedPhaseFailure, match="different phase identity") as raised:
        executor(context)

    assert raised.value.failure_class == "authority_identity_mismatch"
    assert calls == []


def test_authority_checker_must_return_explicit_supported_verdict(tmp_path: Path) -> None:
    executor = SyntheticPhaseExecutor(
        authority_checker=lambda received: 1,
        runner=lambda received: _inventory(received),
    )

    with pytest.raises(TypedPhaseFailure) as raised:
        executor.execute(_context(tmp_path))

    assert raised.value.failure_class == "invalid_authority_verdict"


def test_full_context_requires_committed_run_full_proxy_authorization(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires proxy authorization"):
        replace(_context(tmp_path), phase="full", phase_execution_id="phase-full-0001")

    with pytest.raises(ValueError, match="RUN_FULL"):
        replace(
            _context(tmp_path, phase="full"),
            proxy_authorization={
                "event_id": "event:proxy-rejected",
                "outcome_hash": "e" * 64,
                "decision": "RETURN_S2",
            },
        )


def test_c2c_shells_accept_only_their_authoritative_phase(tmp_path: Path) -> None:
    checker = lambda context: None
    runner = lambda context: _inventory(context)

    with pytest.raises(TypedPhaseFailure) as proxy_failure:
        C2CProxyPhaseExecutor(checker, runner).execute(_context(tmp_path, phase="full"))
    with pytest.raises(TypedPhaseFailure) as full_failure:
        C2CFullPhaseExecutor(checker, runner).execute(_context(tmp_path))

    assert proxy_failure.value.failure_class == "phase_mismatch"
    assert full_failure.value.failure_class == "phase_mismatch"


def test_inventory_is_bound_to_exact_phase_identity(tmp_path: Path) -> None:
    context = _context(tmp_path)
    other = replace(context, phase_execution_id="phase-proxy-0002")
    executor = SyntheticPhaseExecutor(
        authority_checker=lambda received: received,
        runner=lambda received: _inventory(other),
    )

    with pytest.raises(TypedPhaseFailure) as raised:
        executor.execute(context)

    assert raised.value.failure_class == "artifact_identity_mismatch"


def test_inventory_rejects_duplicate_kind_wrong_producer_and_unsafe_path(tmp_path: Path) -> None:
    context = _context(tmp_path)
    valid = {
        "kind": "proxy_results",
        "source_path": "runner/proxy.json",
        "producer_run_id": context.producer_run_id,
    }
    with pytest.raises(ValueError, match="duplicate artifact kind"):
        PhaseArtifactInventory(context, [valid, valid])
    with pytest.raises(ValueError, match="producer_run_id"):
        PhaseArtifactInventory(context, [{**valid, "producer_run_id": "producer-other-0001"}])
    with pytest.raises(ValueError, match="safe project-relative"):
        PhaseArtifactInventory(context, [{**valid, "source_path": "../proxy.json"}])


def test_inventory_manifest_matches_phase_execution_schema_surface(tmp_path: Path) -> None:
    context = _context(tmp_path)
    manifest = _inventory(context, kind="proxy_results").to_manifest()

    assert manifest["schema_version"] == "auto_research_phase_execution_manifest_v1"
    assert manifest["attempt_id"] == context.attempt_id
    assert manifest["phase_execution_id"] == context.phase_execution_id
    assert manifest["phase_start_event_id"] == context.phase_start_event_id
    assert manifest["artifacts"][0]["producer_run_id"] == context.producer_run_id


def test_untyped_runner_exception_becomes_typed_phase_failure(tmp_path: Path) -> None:
    def runner(context: AuthoritativePhaseContext) -> PhaseArtifactInventory:
        raise OSError("command unavailable")

    executor = GenericExternalPhaseExecutor(authority_checker=lambda context: True, runner=runner)

    with pytest.raises(TypedPhaseFailure) as raised:
        executor.execute(_context(tmp_path))

    assert raised.value.failure_class == "phase_execution_failed"
    assert raised.value.retryable is True
    assert raised.value.details["exception_type"] == "OSError"
