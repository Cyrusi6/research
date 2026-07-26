from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

import auto_research.agents.experiment as experiment_module
import auto_research.derivation_validation as derivation_validation
from auto_research.agents.experiment import ExperimentAgent
from auto_research.command_journal import LedgerCommandJournal
from auto_research.contract_store import ContractStore, canonical_contract_bytes, validate_schema
from auto_research.derivation_validation import DerivedEvidence
from auto_research.research_state import IntegrityError, ResearchEventLedger
from auto_research.s3_validation import validate_trial_precommit
from auto_research.validators import run_stage_gate
from support.m1152_local_subprocess import create_generic_project, generic_invocations


@dataclass(frozen=True)
class _CompletedProject:
    root: Path
    config: dict[str, Any]


@pytest.fixture(scope="module")
def completed_generic_source(tmp_path_factory: pytest.TempPathFactory) -> _CompletedProject:
    root = (tmp_path_factory.mktemp("m11531-decoder-baseline") / "project").resolve()
    root.mkdir()
    context, _, _ = create_generic_project(root)
    result = ExperimentAgent(context).run()
    assert (result.get("route_outcome") or {}).get("next_action") == "PROPOSE_NEXT_VARIANT", result
    _assert_complete_authority_baseline(root, context.config)
    return _CompletedProject(root=root, config=deepcopy(context.config))


@pytest.fixture
def completed_generic_project(
    tmp_path: Path,
    completed_generic_source: _CompletedProject,
) -> _CompletedProject:
    root = (tmp_path / "completed-copy" / completed_generic_source.root.name).resolve()
    shutil.copytree(completed_generic_source.root, root)
    return _CompletedProject(root=root, config=deepcopy(completed_generic_source.config))


def test_same_id_version_runtime_decoder_drift_cannot_reinterpret_frozen_plan(
    completed_generic_project: _CompletedProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = completed_generic_project.root, completed_generic_project.config
    baseline = _assert_complete_authority_baseline(root, config)
    attempt = baseline["attempt"]
    phase = baseline["trial_result"]["completeness"]
    phase_contract = next(
        item for item in attempt["frozen_trial_spec"]["phase_contracts"] if item["phase"] == phase
    )
    descriptor = deepcopy(phase_contract["derivation_plan"]["decoder_descriptor"])
    decoder_key = (descriptor["decoder_id"], descriptor["decoder_version"])
    immutable_bytes_before = ContractStore(root).read_bytes(descriptor["immutable_ref"])
    marker = root / "runner" / "m11531-runtime-decoder-drift.jsonl"
    original_decoder = derivation_validation._DECODER_REGISTRY[decoder_key]

    def drifting_decoder(*args: Any, **kwargs: Any) -> tuple[DerivedEvidence, ...]:
        outputs = list(original_decoder(*args, **kwargs))
        _append_jsonl(
            marker,
            {
                "decoder_id": descriptor["decoder_id"],
                "decoder_version": descriptor["decoder_version"],
                "implementation_hash": descriptor["implementation_hash"],
            },
        )
        target_index = next(index for index, output in enumerate(outputs) if output.kind == "main_results")
        target = outputs[target_index]
        payload = json.loads(target.raw_bytes.decode("utf-8"))
        candidate = next(row for row in payload["rows"] if row["role"] == "candidate")
        candidate["metric_value"] = float(candidate["metric_value"]) + 0.125
        outputs[target_index] = DerivedEvidence(
            output_id=target.output_id,
            kind=target.kind,
            schema_version=target.schema_version,
            raw_bytes=canonical_contract_bytes(payload),
        )
        return tuple(outputs)

    monkeypatch.setitem(derivation_validation._DECODER_REGISTRY, decoder_key, drifting_decoder)
    outcomes = {
        "precommit": _capture_authority_call(
            lambda: validate_trial_precommit(
                project_root=root,
                direction=baseline["direction"],
                variant=baseline["variant"],
                attempt=attempt,
                trial_spec=attempt["frozen_trial_spec"],
                trial_result=baseline["trial_result"],
                state=baseline["state"],
            )
        ),
        "state": _capture_authority_call(lambda: ResearchEventLedger(root).state()),
        "rebuild": _capture_authority_call(lambda: ResearchEventLedger(root).rebuild()),
        "query": _capture_authority_call(
            lambda: ResearchEventLedger(root).query_operation_result(baseline["final_event_id"])
        ),
        "gate": _capture_authority_call(
            lambda: _require_gate_pass(run_stage_gate("S3_experiment", root, config).to_dict())
        ),
    }

    assert ContractStore(root).read_bytes(descriptor["immutable_ref"]) == immutable_bytes_before
    assert phase_contract["derivation_plan"]["decoder_descriptor"] == descriptor
    drift_invocations = _jsonl_records(marker)
    assert drift_invocations == [], (
        "authority readers executed the mutable runtime registry implementation for a frozen "
        "decoder ID/version instead of using or authenticating the immutable implementation "
        f"artifact first; decoder={decoder_key} descriptor={descriptor} "
        f"drift_invocations={drift_invocations} outcomes={outcomes}"
    )


def test_derive_execution_occurs_only_inside_journal_runner(
    tmp_path: Path,
    completed_generic_source: _CompletedProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert completed_generic_source.root.is_dir()
    root, context = _new_generic_project(tmp_path, "journal-bound-derive")
    marker = root / "runner" / "m11531-producing-derive.jsonl"
    journal_depth = {"value": 0}
    original_derive = experiment_module.derive_evidence_deterministically
    original_run_once = LedgerCommandJournal.run_once

    def tracked_derive(*args: Any, **kwargs: Any):
        plan = kwargs["derivation_plan"]
        _append_jsonl(
            marker,
            {
                "plan_id": plan["plan_id"],
                "phase": kwargs["phase"],
                "inside_journal_runner": journal_depth["value"] > 0,
            },
        )
        return original_derive(*args, **kwargs)

    def tracked_run_once(self: LedgerCommandJournal, *args: Any, **kwargs: Any):
        is_derive = str(kwargs.get("command_spec_id") or "").endswith("derive-evidence")
        if is_derive:
            journal_depth["value"] += 1
        try:
            return original_run_once(self, *args, **kwargs)
        finally:
            if is_derive:
                journal_depth["value"] -= 1

    monkeypatch.setattr(experiment_module, "derive_evidence_deterministically", tracked_derive)
    monkeypatch.setattr(LedgerCommandJournal, "run_once", tracked_run_once)

    result = ExperimentAgent(context).run()
    assert (result.get("route_outcome") or {}).get("next_action") == "PROPOSE_NEXT_VARIANT", result
    assert run_stage_gate("S3_experiment", root, context.config).to_dict()["status"] == "PASS"
    invocations = _jsonl_records(marker)
    assert invocations and all(item["inside_journal_runner"] is True for item in invocations), (
        "the producing decoder ran before LedgerCommandJournal.run_once performed its historical "
        f"command/receipt check; invocations={invocations}"
    )


def test_durable_derive_receipt_recovery_does_not_reinvoke_producing_decoder(
    tmp_path: Path,
    completed_generic_source: _CompletedProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert completed_generic_source.root.is_dir()
    root, context = _new_generic_project(tmp_path, "durable-derive-recovery")
    marker = root / "runner" / "m11531-producing-derive.jsonl"
    original_derive = experiment_module.derive_evidence_deterministically
    original_complete = ResearchEventLedger.complete_phase_command
    crashed = {"value": False}

    def tracked_derive(*args: Any, **kwargs: Any):
        _append_jsonl(
            marker,
            {
                "plan_id": kwargs["derivation_plan"]["plan_id"],
                "phase": kwargs["phase"],
            },
        )
        return original_derive(*args, **kwargs)

    def crash_after_durable_receipt(
        self: ResearchEventLedger,
        command_id: str,
        receipt_ref: dict[str, Any],
    ) -> dict[str, Any]:
        record = self.phase_command(command_id) or {}
        command = record.get("command") or {}
        if str(command.get("command_spec_id") or "").endswith("derive-evidence") and not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("m11531 crash after durable derive receipt before Completed")
        return original_complete(self, command_id, receipt_ref)

    monkeypatch.setattr(experiment_module, "derive_evidence_deterministically", tracked_derive)
    monkeypatch.setattr(ResearchEventLedger, "complete_phase_command", crash_after_durable_receipt)
    with pytest.raises(RuntimeError, match="after durable derive receipt before Completed"):
        ExperimentAgent(context).run()
    monkeypatch.setattr(ResearchEventLedger, "complete_phase_command", original_complete)

    ledger = ResearchEventLedger(root)
    derive_record = _single_derive_record(ledger.state(), status="started")
    durable_receipt_ref = _receipt_locator_reference(root, derive_record["command"])
    ContractStore(root).verify(durable_receipt_ref)
    physical_before = generic_invocations(root)
    producing_before = _jsonl_records(marker)
    assert len(producing_before) == 1, producing_before

    restarted = ExperimentAgent(context).run()
    assert (restarted.get("route_outcome") or {}).get("next_action") == "PROPOSE_NEXT_VARIANT", restarted
    assert run_stage_gate("S3_experiment", root, context.config).to_dict()["status"] == "PASS"
    producing_after = _jsonl_records(marker)
    physical_after = generic_invocations(root)
    state = ResearchEventLedger(root).state()
    completed = _single_derive_record(state, status="completed")

    assert physical_after == physical_before
    assert completed["receipt_ref"] == durable_receipt_ref
    assert producing_after == producing_before, (
        "recovery found a trustworthy durable derive receipt but invoked the producing decoder "
        f"again before journal reconciliation; before={producing_before} after={producing_after}"
    )


def test_phase_command_completed_rejects_hash_consistent_semantically_wrong_derivation(
    tmp_path: Path,
    completed_generic_source: _CompletedProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert completed_generic_source.root.is_dir()
    root, context = _new_generic_project(tmp_path, "semantic-completed-attack")
    _crash_before_derive_completed(context, monkeypatch)
    ledger = ResearchEventLedger(root)
    derive_record = _single_derive_record(ledger.state(), status="started")
    valid_receipt_ref = _receipt_locator_reference(root, derive_record["command"])

    valid_root = (tmp_path / "valid-derive-completion" / root.name).resolve()
    shutil.copytree(root, valid_root)
    valid_ledger = ResearchEventLedger(valid_root)
    valid_before = len(valid_ledger.events())
    valid_result = valid_ledger.complete_phase_command(
        derive_record["command"]["command_id"],
        valid_receipt_ref,
    )
    assert valid_result["status"] == "completed"
    assert len(valid_ledger.events()) == valid_before + 1

    attacked_receipt_ref, attack_facts = _hash_consistent_wrong_derivation_receipt(
        root,
        derive_record,
        valid_receipt_ref,
    )
    events_before = ledger.events()
    error: IntegrityError | None = None
    try:
        ledger.complete_phase_command(
            derive_record["command"]["command_id"],
            attacked_receipt_ref,
        )
    except IntegrityError as exc:
        error = exc
    events_after = ledger.events()

    assert error is not None and any(
        token in str(error).lower() for token in ("deterministic", "normalized", "derivation")
    ) and events_after == events_before, (
        "PhaseCommandCompleted accepted a schema-valid, hash-consistent derive receipt whose "
        "normalized metric bytes do not match deterministic recomputation from the frozen raw "
        f"receipts; error={error!r} events={len(events_before)}->{len(events_after)} "
        f"attack={attack_facts}"
    )


def _new_generic_project(tmp_path: Path, name: str) -> tuple[Path, Any]:
    root = (tmp_path / name / "project").resolve()
    root.parent.mkdir()
    root.mkdir()
    context, _, _ = create_generic_project(root)
    return root, context


def _assert_complete_authority_baseline(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    ledger = ResearchEventLedger(root)
    state = ledger.state()
    assert len(state["attempts"]) == 1
    attempt = next(iter(state["attempts"].values()))
    trial_result = state["trial_results"][attempt["attempt_id"]]
    direction = state["directions"][attempt["direction_semantic_hash"]]["spec"]
    variant = state["variants"][attempt["variant_spec_hash"]]
    final_events = [
        event
        for event in ledger.events()
        if event["event_type"] == "AttemptFinalized"
        and event["payload"]["trial_result"]["attempt_id"] == attempt["attempt_id"]
    ]
    assert len(final_events) == 1
    precommit = validate_trial_precommit(
        project_root=root,
        direction=direction,
        variant=variant,
        attempt=attempt,
        trial_spec=attempt["frozen_trial_spec"],
        trial_result=trial_result,
        state=state,
    )
    assert precommit["status"] == "PASS"
    rebuilt = ledger.rebuild()
    assert rebuilt["last_sequence"] == state["last_sequence"]
    queried = ledger.query_operation_result(final_events[0]["event_id"])
    assert queried["trial_result"] == trial_result
    gate = run_stage_gate("S3_experiment", root, config).to_dict()
    assert gate["status"] == "PASS", gate
    return {
        "state": state,
        "attempt": attempt,
        "trial_result": trial_result,
        "direction": direction,
        "variant": variant,
        "final_event_id": final_events[0]["event_id"],
    }


def _capture_authority_call(operation: Callable[[], Any]) -> str:
    try:
        result = operation()
    except BaseException as exc:
        return f"{type(exc).__name__}: {exc}"
    if isinstance(result, dict) and "status" in result:
        return str(result["status"])
    return "PASS"


def _require_gate_pass(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("status") != "PASS":
        raise IntegrityError(f"Gate rejected frozen decoder authority: {report}")
    return report


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _single_derive_record(state: dict[str, Any], *, status: str) -> dict[str, Any]:
    matches = [
        record
        for record in state["phase_commands"].values()
        if record["status"] == status
        and str(record["command"]["command_spec_id"]).endswith("derive-evidence")
    ]
    assert len(matches) == 1, matches
    return matches[0]


def _receipt_locator_reference(root: Path, command: dict[str, Any]) -> dict[str, Any]:
    locator = LedgerCommandJournal(root, ResearchEventLedger(root))._locator_path(command)
    payload = json.loads(locator.read_text(encoding="utf-8"))
    assert payload["command_id"] == command["command_id"]
    assert payload["command_hash"] == command["command_hash"]
    reference = payload["receipt_ref"]
    ContractStore(root).verify(reference)
    return reference


def _crash_before_derive_completed(context: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    original_complete = ResearchEventLedger.complete_phase_command
    crashed = {"value": False}

    def crash(
        self: ResearchEventLedger,
        command_id: str,
        receipt_ref: dict[str, Any],
    ) -> dict[str, Any]:
        record = self.phase_command(command_id) or {}
        command = record.get("command") or {}
        if str(command.get("command_spec_id") or "").endswith("derive-evidence") and not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("m11531 hold derive receipt before Completed")
        return original_complete(self, command_id, receipt_ref)

    monkeypatch.setattr(ResearchEventLedger, "complete_phase_command", crash)
    with pytest.raises(RuntimeError, match="hold derive receipt before Completed"):
        ExperimentAgent(context).run()
    monkeypatch.setattr(ResearchEventLedger, "complete_phase_command", original_complete)


def _hash_consistent_wrong_derivation_receipt(
    root: Path,
    derive_record: dict[str, Any],
    valid_receipt_ref: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    store = ContractStore(root)
    valid_receipt = store.read_json(
        valid_receipt_ref,
        schema_file="phase_run_receipt_v5.schema.json",
    )
    valid_manifest = store.read_json(
        valid_receipt["derivation_ref"],
        schema_file="evidence_derivation_manifest_v3.schema.json",
    )
    output_index = next(
        index for index, output in enumerate(valid_manifest["normalized_outputs"])
        if output["kind"] == "main_results"
    )
    valid_output = valid_manifest["normalized_outputs"][output_index]
    valid_payload = store.read_json(
        valid_output["contract_ref"],
        schema_file="main_results_v3.schema.json",
    )
    attacked_payload = deepcopy(valid_payload)
    candidate = next(row for row in attacked_payload["rows"] if row["role"] == "candidate")
    candidate["metric_value"] = float(candidate["metric_value"]) + 7.0
    attacked_output_ref = store.put_json(
        attacked_payload,
        schema_file="main_results_v3.schema.json",
    )

    attacked_manifest = deepcopy(valid_manifest)
    attacked_manifest["normalized_outputs"][output_index]["contract_ref"] = deepcopy(attacked_output_ref)
    attacked_manifest["normalized_outputs"][output_index]["content_hash"] = attacked_output_ref["digest"]
    attacked_manifest_ref = store.put_json(
        attacked_manifest,
        schema_file="evidence_derivation_manifest_v3.schema.json",
    )

    attacked_receipt = deepcopy(valid_receipt)
    receipt_output = next(
        output for output in attacked_receipt["outputs"] if output["output_id"] == valid_output["output_id"]
    )
    receipt_output["contract_ref"] = deepcopy(attacked_output_ref)
    receipt_output["content_hash"] = attacked_output_ref["digest"]
    attacked_receipt["derivation_ref"] = deepcopy(attacked_manifest_ref)
    attacked_receipt["derivation_hash"] = attacked_manifest_ref["digest"]
    validate_schema(attacked_receipt, "phase_run_receipt_v5.schema.json")
    attacked_receipt_ref = store.put_json(
        attacked_receipt,
        schema_file="phase_run_receipt_v5.schema.json",
    )

    store.verify(attacked_output_ref)
    store.verify(attacked_manifest_ref)
    store.verify(attacked_receipt_ref)
    assert attacked_manifest["source_commands"] == valid_manifest["source_commands"]
    assert attacked_manifest["decoder_descriptor"] == valid_manifest["decoder_descriptor"]
    assert attacked_output_ref["digest"] != valid_output["content_hash"]
    assert attacked_receipt["command_id"] == derive_record["command"]["command_id"]
    return attacked_receipt_ref, {
        "valid_receipt_hash": valid_receipt_ref["digest"],
        "attacked_receipt_hash": attacked_receipt_ref["digest"],
        "valid_derivation_hash": valid_receipt["derivation_hash"],
        "attacked_derivation_hash": attacked_manifest_ref["digest"],
        "valid_normalized_hash": valid_output["content_hash"],
        "attacked_normalized_hash": attacked_output_ref["digest"],
    }
