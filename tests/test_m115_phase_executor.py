from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from auto_research.phase_execution import (
    AuthoritativePhaseContext,
    C2CFullPhaseExecutor,
    C2CProxyPhaseExecutor,
    PhaseArtifactInventory,
    PhaseAuthorization,
    ResearchLedgerPhaseAuthority,
    SyntheticPhaseExecutor,
    GenericExternalPhaseExecutor,
    TypedPhaseFailure,
)
from auto_research.contract_store import ContractStore
from auto_research.phase_command_plan import build_phase_command_plan, store_phase_command_plan
from auto_research.agents.experiment import ExperimentAgent
from auto_research.research_state import IntegrityError


def _authorization(*, phase: str = "proxy") -> PhaseAuthorization:
    full = phase == "full"
    return PhaseAuthorization(
        attempt_id="attempt-0001",
        lifecycle_generation=2,
        phase=phase,
        phase_execution_id=f"phase-{phase}-0001",
        phase_start_event_id=f"event:{phase}-started",
        phase_start_event_hash="a" * 64,
        phase_start_sequence=3,
        producer_run_id=f"producer-{phase}-0001",
        implementation_hash="6" * 64,
        attempt_input_hash="7" * 64,
        trial_spec_hash="5" * 64,
        command_plan_hash="8" * 64,
        phase_contract_hash="9" * 64,
        expected_evidence_kinds=("proxy_results",) if phase == "proxy" else ("main_results",),
        adapter_identity="adapter-c2c-001",
        provenance_mode="local_external",
        state="FULL_RUNNING" if full else "PROXY_RUNNING",
        proxy_commit_event_id="event:proxy-committed" if full else None,
        proxy_commit_event_hash="b" * 64 if full else None,
        proxy_outcome_hash="c" * 64 if full else None,
    )


def _context(tmp_path: Path, *, phase: str = "proxy") -> AuthoritativePhaseContext:
    authorization = _authorization(phase=phase)
    attempt = {
        "attempt_id": authorization.attempt_id,
        "direction_semantic_hash": "1" * 64,
        "direction_spec_hash": "2" * 64,
        "variant_semantic_hash": "3" * 64,
        "variant_spec_hash": "4" * 64,
        "trial_spec_hash": authorization.trial_spec_hash,
        "lifecycle_generation": authorization.lifecycle_generation,
        "implementation_hash": authorization.implementation_hash,
        "attempt_input_hash": authorization.attempt_input_hash,
    }
    return AuthoritativePhaseContext.from_authorization(tmp_path, attempt, authorization)


def _inventory(context: AuthoritativePhaseContext) -> PhaseArtifactInventory:
    kind = context.expected_evidence_kinds[0]
    return PhaseArtifactInventory(
        context,
        ({
            "kind": kind,
            "source_path": f"experiment/attempts/{context.attempt_id}/{kind}.json",
            "content_hash": "d" * 64,
            "receipt_hash": "e" * 64,
            "producer_run_id": context.producer_run_id,
        },),
    )


def _coverage() -> dict:
    return {
        "mode": "exact_cartesian",
        "datasets": ["fixture-dataset"],
        "seeds": [7],
        "metrics": ["score"],
        "roles": ["baseline", "candidate"],
    }


def _proxy_command(tmp_path: Path) -> dict:
    return {
        "argv": ["true"],
        "cwd": str(tmp_path),
        "physical_raw_outputs": [{
            "output_id": "raw-proxy-results",
            "kind": "raw_proxy_results",
            "schema_version": "auto_research_proxy_results_v1",
            "locator": "runner/proxy-results.json",
            "locator_type": "file",
            "dataset_id": None,
            "role": None,
            "required": True,
            "normalized_kinds": ["proxy_results"],
        }],
    }


class Authority:
    def __init__(self, authorization):
        self.authorization = authorization
        self.calls = 0

    def authorize_phase(self, context):
        self.calls += 1
        return self.authorization


def test_executor_rechecks_exact_sqlite_authorization_before_and_after_runner(tmp_path: Path) -> None:
    context = _context(tmp_path)
    plan = build_phase_command_plan(
        phase="proxy",
        adapter_id="c2c-phase-adapter",
        adapter_version="1",
        provenance_mode="production",
        variant_spec_hash=context.variant_spec_hash,
        source_snapshot_hash=context.implementation_hash,
        command_values=(_proxy_command(tmp_path),),
        expected_evidence=({"kind": "proxy_results", "schema_version": "auto_research_proxy_results_v1", "required": True},),
        default_cwd=str(tmp_path),
        project_root=tmp_path,
        coverage_contract=_coverage(),
    )
    _, plan_hash = store_phase_command_plan(tmp_path, plan)
    authorization = replace(_authorization(), command_plan_hash=plan_hash)
    context = replace(context, command_plan_hash=plan_hash, authorization_hash=authorization.authorization_hash)
    authority = Authority(authorization)
    calls: list[str] = []
    executor = C2CProxyPhaseExecutor(authority, lambda received: calls.append("runner") or _inventory(received))
    assert executor.execute(context).context == context
    assert calls == ["runner"]
    assert authority.calls == 2


def test_executor_kind_must_match_authorized_adapter(tmp_path: Path) -> None:
    context = _context(tmp_path)
    plan = build_phase_command_plan(
        phase="proxy",
        adapter_id="c2c-phase-adapter",
        adapter_version="1",
        provenance_mode="production",
        variant_spec_hash=context.variant_spec_hash,
        source_snapshot_hash=context.implementation_hash,
        command_values=(_proxy_command(tmp_path),),
        expected_evidence=({"kind": "proxy_results", "schema_version": "auto_research_proxy_results_v1", "required": True},),
        default_cwd=str(tmp_path),
        project_root=tmp_path,
        coverage_contract=_coverage(),
    )
    _, plan_hash = store_phase_command_plan(tmp_path, plan)
    authorization = replace(_authorization(), command_plan_hash=plan_hash)
    context = replace(context, command_plan_hash=plan_hash, authorization_hash=authorization.authorization_hash)
    calls: list[str] = []
    with pytest.raises(TypedPhaseFailure, match="not authorized"):
        GenericExternalPhaseExecutor(
            Authority(authorization),
            lambda received: calls.append("runner") or _inventory(received),
        ).execute(context)
    assert calls == []


def test_synthetic_executor_requires_synthetic_authorization(tmp_path: Path) -> None:
    context = _context(tmp_path)
    calls: list[str] = []
    with pytest.raises(TypedPhaseFailure, match="not authorized"):
        SyntheticPhaseExecutor(
            Authority(_authorization()),
            lambda received: calls.append("runner") or _inventory(received),
        ).execute(context)
    assert calls == []


@pytest.mark.parametrize("verdict", [True, None, False, {"state": "PROXY_RUNNING"}])
def test_bool_none_or_mapping_authority_is_fail_closed(tmp_path: Path, verdict) -> None:
    context = _context(tmp_path)
    calls: list[str] = []

    class Forged:
        def authorize_phase(self, received):
            return verdict

    with pytest.raises(TypedPhaseFailure, match="PhaseAuthorization"):
        C2CProxyPhaseExecutor(Forged(), lambda received: calls.append("runner") or _inventory(received)).execute(context)
    assert calls == []


def test_authorization_generation_drift_blocks_runner(tmp_path: Path) -> None:
    context = _context(tmp_path)
    drifted = replace(_authorization(), lifecycle_generation=3)
    calls: list[str] = []
    with pytest.raises(TypedPhaseFailure, match="differs"):
        SyntheticPhaseExecutor(Authority(drifted), lambda received: calls.append("runner") or _inventory(received)).execute(context)
    assert calls == []


def test_full_requires_committed_proxy_authorization(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="committed proxy"):
        replace(
            _authorization(phase="full"),
            proxy_commit_event_id=None,
            proxy_commit_event_hash=None,
            proxy_outcome_hash=None,
        )


def test_phase_specific_executor_rejects_wrong_phase_before_runner(tmp_path: Path) -> None:
    context = _context(tmp_path, phase="full")
    calls: list[str] = []
    with pytest.raises(TypedPhaseFailure, match="requires proxy"):
        C2CProxyPhaseExecutor(Authority(_authorization(phase="full")), lambda received: calls.append("runner") or _inventory(received)).execute(context)
    assert calls == []


def test_inventory_requires_exact_authorized_kinds_and_receipt_hash(tmp_path: Path) -> None:
    context = _context(tmp_path)
    with pytest.raises(ValueError, match="exactly"):
        PhaseArtifactInventory(context, ())
    with pytest.raises(ValueError, match="not authorized"):
        PhaseArtifactInventory(context, ({
            "kind": "main_results", "source_path": "safe/main.json", "content_hash": "d" * 64,
            "receipt_hash": "e" * 64, "producer_run_id": context.producer_run_id,
        },))


def test_research_ledger_authority_reconstructs_exact_v3_authorization(tmp_path: Path) -> None:
    expected = _authorization()
    context = _context(tmp_path)
    command_plan_ref = ContractStore(tmp_path).put_bytes(b'{"command_plan":"fixture"}')
    manifest = {
        "schema_version": "auto_research_phase_execution_manifest_v3",
        "attempt_id": expected.attempt_id,
        "direction_semantic_hash": "1" * 64,
        "direction_spec_hash": "2" * 64,
        "variant_semantic_hash": "3" * 64,
        "variant_spec_hash": "4" * 64,
        "trial_spec_hash": expected.trial_spec_hash,
        "lifecycle_generation": expected.lifecycle_generation,
        "implementation_hash": expected.implementation_hash,
        "attempt_input_hash": expected.attempt_input_hash,
        "phase": expected.phase,
        "phase_execution_id": expected.phase_execution_id,
        "phase_start_event_id": expected.phase_start_event_id,
        "producer_run_id": expected.producer_run_id,
        "command_plan_hash": expected.command_plan_hash,
        "phase_contract_hash": expected.phase_contract_hash,
        "expected_evidence_kinds": list(expected.expected_evidence_kinds),
        "provenance_mode": "local-external",
        "proxy_evaluation_binding": None,
        "proxy_authorization": None,
        "command_plan_ref": command_plan_ref,
    }

    class Ledger:
        def state(self):
            return {"attempts": {expected.attempt_id: {
                "attempt_id": expected.attempt_id, "lifecycle_generation": expected.lifecycle_generation,
                "implementation_hash": expected.implementation_hash, "attempt_input_hash": expected.attempt_input_hash,
                "trial_spec_hash": expected.trial_spec_hash, "state": expected.state,
                "frozen_trial_spec": {"execution_contract": {"runtime_config": {"collector": "c2c-001"}}},
                "phase_executions": {"proxy": {"phase_start_event_id": expected.phase_start_event_id}},
            }}}

        def events(self):
            return [{
                "event_id": expected.phase_start_event_id, "event_hash": expected.phase_start_event_hash,
                "sequence": expected.phase_start_sequence, "event_type": "ProxyPhaseStarted",
                "payload": {"phase_execution_manifest": manifest},
            }]

    assert ResearchLedgerPhaseAuthority(Ledger()).authorize_phase(context) == expected


def test_research_ledger_authority_rejects_replaced_manifest(tmp_path: Path) -> None:
    expected = _authorization()

    class Ledger:
        def state(self):
            return {"attempts": {expected.attempt_id: {
                "attempt_id": expected.attempt_id, "lifecycle_generation": expected.lifecycle_generation,
                "implementation_hash": expected.implementation_hash, "attempt_input_hash": expected.attempt_input_hash,
                "trial_spec_hash": expected.trial_spec_hash, "state": expected.state,
                "phase_executions": {"proxy": {"phase_start_event_id": expected.phase_start_event_id}},
            }}}

        def events(self):
            return [{"event_id": expected.phase_start_event_id, "event_hash": "a" * 64, "sequence": 3,
                     "event_type": "ProxyPhaseStarted", "payload": {"phase_execution_manifest": {"schema_version": "auto_research_phase_execution_manifest_v1"}}}]

    with pytest.raises(ValueError, match="v3"):
        ResearchLedgerPhaseAuthority(Ledger()).authorize_phase(_context(tmp_path))


def test_experiment_side_effect_without_ledger_context_is_fail_closed(tmp_path: Path) -> None:
    agent = object.__new__(ExperimentAgent)
    agent._active_phase_context = None
    agent._active_command_journal = None
    calls: list[str] = []

    class Runner:
        def run_step(self, **kwargs):
            calls.append("runner")
            return {"status": "ok", "returncode": 0}

    agent.runner = Runner()
    with pytest.raises(IntegrityError, match="authoritative phase context"):
        agent._run_authoritative_step(
            name="full-train",
            command={"argv": ["true"], "cwd": str(tmp_path)},
            cwd=tmp_path,
        )
    assert calls == []


def test_phase_agnostic_c2c_production_entry_is_removed() -> None:
    assert not hasattr(ExperimentAgent, "_run_single_c2c_candidate")
