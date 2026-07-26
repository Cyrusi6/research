from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

from auto_research.agents.base import AgentContext
from auto_research.agents.experiment import ExperimentAgent
from auto_research.artifacts import ArtifactManager
from auto_research.command_journal import LedgerCommandJournal
from auto_research.contract_store import ContractStore, validate_schema
from auto_research.derivation_contracts import freeze_decoder_descriptor
from auto_research.evidence_lineage import validate_receipt_bound_evidence
from auto_research.llm import ModelClient
from auto_research.research_state import IntegrityError, ResearchEventLedger
from auto_research.validators import run_stage_gate
from support.m1152_local_subprocess import c2c_invocations, create_c2c_project


@dataclass(frozen=True)
class DerivationAttackBaseline:
    started_root: Path
    completed_root: Path
    config: dict[str, Any]
    phase: str
    derive_command_id: str
    valid_receipt_ref: dict[str, Any]


@dataclass(frozen=True)
class DerivationCandidate:
    receipt_ref: dict[str, Any]
    receipt: dict[str, Any]
    manifest_ref: dict[str, Any]
    manifest: dict[str, Any]
    facts: dict[str, Any]


def build_derivation_attack_baseline(base: Path) -> DerivationAttackBaseline:
    monkeypatch = pytest.MonkeyPatch()
    original_complete = ResearchEventLedger.complete_phase_command
    crashed = {"value": False}
    try:
        root, _, context, _ = create_c2c_project(
            base,
            monkeypatch,
            profile="standard",
            proxy_accuracy=0.49,
            name="m11531-hash-consistent-started",
        )

        def hold_derive_receipt(
            ledger: ResearchEventLedger,
            command_id: str,
            receipt_ref: dict[str, Any],
        ) -> dict[str, Any]:
            record = ledger.phase_command(command_id) or {}
            command = record.get("command") or {}
            if (
                str(command.get("command_spec_id") or "").endswith("derive-evidence")
                and not crashed["value"]
            ):
                crashed["value"] = True
                raise RuntimeError("m11531 hold durable derive receipt before Completed")
            return original_complete(ledger, command_id, receipt_ref)

        monkeypatch.setattr(ResearchEventLedger, "complete_phase_command", hold_derive_receipt)
        with pytest.raises(RuntimeError, match="hold durable derive receipt before Completed"):
            ExperimentAgent(context).run()
        monkeypatch.setattr(ResearchEventLedger, "complete_phase_command", original_complete)

        ledger = ResearchEventLedger(root)
        derive_record = single_derive_record(ledger.state(), status="started")
        valid_receipt_ref = receipt_locator_reference(root, derive_record["command"])
        physical_before = c2c_invocations(root)

        completed_root = (base / "completed" / root.name).resolve()
        completed_root.parent.mkdir()
        shutil.copytree(root, completed_root, symlinks=True)
        completed_ledger = ResearchEventLedger(completed_root)
        before_events = len(completed_ledger.events())
        completed = completed_ledger.complete_phase_command(
            derive_record["command"]["command_id"],
            valid_receipt_ref,
        )
        assert completed["status"] == "completed"
        assert len(completed_ledger.events()) == before_events + 1
        result = ExperimentAgent(agent_context(completed_root, context.config)).run()
        assert (result.get("route_outcome") or {}).get("next_action") == "PROPOSE_NEXT_VARIANT", result
        assert c2c_invocations(completed_root) == physical_before
        gate = run_stage_gate("S3_experiment", completed_root, context.config).to_dict()
        assert gate["status"] == "PASS", gate
        assert completed_ledger.rebuild() == completed_ledger.state()
        return DerivationAttackBaseline(
            started_root=root,
            completed_root=completed_root,
            config=deepcopy(context.config),
            phase=str(derive_record["command"]["phase"]),
            derive_command_id=str(derive_record["command"]["command_id"]),
            valid_receipt_ref=deepcopy(valid_receipt_ref),
        )
    finally:
        monkeypatch.undo()


def agent_context(root: Path, config: Mapping[str, Any]) -> AgentContext:
    copied = deepcopy(dict(config))
    return AgentContext(
        root,
        copied,
        ArtifactManager(root),
        ModelClient(copied, project_root=root),
    )


def clone_project(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True)
    root = (destination / source.name).resolve()
    shutil.copytree(source, root, symlinks=True)
    return root


def single_derive_record(state: Mapping[str, Any], *, status: str) -> dict[str, Any]:
    matches = [
        record
        for record in state["phase_commands"].values()
        if record["status"] == status
        and str(record["command"]["command_spec_id"]).endswith("derive-evidence")
    ]
    assert len(matches) == 1, matches
    return deepcopy(matches[0])


def receipt_locator_reference(root: Path, command: Mapping[str, Any]) -> dict[str, Any]:
    locator = LedgerCommandJournal(root, ResearchEventLedger(root))._locator_path(dict(command))
    payload = json.loads(locator.read_text(encoding="utf-8"))
    assert payload["command_id"] == command["command_id"]
    assert payload["command_hash"] == command["command_hash"]
    reference = payload["receipt_ref"]
    ContractStore(root).verify(reference)
    return deepcopy(reference)


def load_candidate_authority(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    derive_record = single_derive_record(ResearchEventLedger(root).state(), status="started")
    receipt_ref = receipt_locator_reference(root, derive_record["command"])
    store = ContractStore(root)
    receipt = store.read_json(receipt_ref, schema_file="phase_run_receipt_v5.schema.json")
    manifest = store.read_json(
        receipt["derivation_ref"],
        schema_file="evidence_derivation_manifest_v3.schema.json",
    )
    return derive_record, receipt, manifest


def build_manifest_candidate(
    root: Path,
    mutate: Callable[[dict[str, Any], dict[str, Any], ContractStore], None],
    *,
    attack: str,
) -> DerivationCandidate:
    derive_record, receipt, manifest = load_candidate_authority(root)
    store = ContractStore(root)
    original_receipt_hash = receipt_locator_reference(root, derive_record["command"])["digest"]
    original_manifest_hash = str(receipt["derivation_hash"])
    mutate(manifest, receipt, store)
    validate_schema(manifest, "evidence_derivation_manifest_v3.schema.json")
    manifest_ref = store.put_json(
        manifest,
        schema_file="evidence_derivation_manifest_v3.schema.json",
    )
    _bind_receipt_to_manifest(receipt, manifest, manifest_ref)
    validate_schema(receipt, "phase_run_receipt_v5.schema.json")
    receipt_ref = store.put_json(receipt, schema_file="phase_run_receipt_v5.schema.json")
    for reference in (manifest_ref, receipt_ref):
        store.verify(reference)
    return DerivationCandidate(
        receipt_ref=deepcopy(receipt_ref),
        receipt=deepcopy(receipt),
        manifest_ref=deepcopy(manifest_ref),
        manifest=deepcopy(manifest),
        facts={
            "attack": attack,
            "derive_command_id": derive_record["command"]["command_id"],
            "valid_receipt_hash": original_receipt_hash,
            "candidate_receipt_hash": receipt_ref["digest"],
            "valid_manifest_hash": original_manifest_hash,
            "candidate_manifest_hash": manifest_ref["digest"],
        },
    )


def reordinal(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = deepcopy(sources)
    for ordinal, source in enumerate(result):
        source["source_ordinal"] = ordinal
    return result


def mutate_self_source(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    del store
    source = deepcopy(manifest["source_commands"][0])
    source.update(
        {
            "command_id": receipt["command_id"],
            "command_spec_id": receipt["command_spec_id"],
            "command_hash": receipt["command_hash"],
            "completed_event_id": receipt["started_event_id"],
        }
    )
    manifest["source_commands"][0] = source


def mutate_missing_source(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    del receipt, store
    assert len(manifest["source_commands"]) >= 2
    manifest["source_commands"] = reordinal(manifest["source_commands"][:-1])


def mutate_extra_source(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    del receipt, store
    source = deepcopy(manifest["source_commands"][0])
    source["command_id"] = "cmd-extra-physical-source"
    source["command_spec_id"] = "proxy-extra-physical-source"
    source["output_id"] = "proxy-extra-physical-output"
    manifest["source_commands"] = reordinal([*manifest["source_commands"], source])


def mutate_duplicate_source(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    del receipt, store
    manifest["source_commands"] = reordinal(
        [*manifest["source_commands"], deepcopy(manifest["source_commands"][0])]
    )


def mutate_reordered_source(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    del receipt, store
    assert len(manifest["source_commands"]) >= 2
    manifest["source_commands"] = reordinal(list(reversed(manifest["source_commands"])))


def mutate_cross_attempt(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    del receipt, store
    manifest["attempt_id"] = "attempt-cross-authority-m11531"


def mutate_cross_phase(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    del receipt, store
    manifest["phase"] = "full" if manifest["phase"] == "proxy" else "proxy"


def mutate_cross_generation(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    del receipt, store
    manifest["lifecycle_generation"] += 1


def mutate_cross_producer(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    del receipt, store
    manifest["producer_run_id"] = "producer-cross-authority-m11531"


def mutate_wrong_source_command_id(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    del receipt, store
    manifest["source_commands"][0]["command_id"] = "cmd-wrong-source-identity"


def mutate_wrong_source_output_id(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    del receipt, store
    manifest["source_commands"][0]["output_id"] = "output-wrong-source-identity"


def mutate_wrong_decoder_artifact(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    del receipt
    descriptor = manifest["decoder_descriptor"]
    bundle = store.read_json(
        descriptor["immutable_ref"],
        schema_file="decoder_implementation_bundle_v1.schema.json",
    )
    program = bundle["decoder_program"]
    semantic_contract = deepcopy(program["semantic_contract"])
    semantic_contract["coverage_contract"]["datasets"] = ["wrong-decoder-dataset"]
    manifest["decoder_descriptor"] = freeze_decoder_descriptor(
        store.project_root,
        decoder_id=program["decoder_id"],
        decoder_version=program["decoder_version"],
        semantic_contract=semantic_contract,
        authority_role_contract=program["authority_role_contract"],
        output_contract=program["output_contract"],
    )


def mutate_wrong_normalized_bytes(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    target = next(
        output
        for output in manifest["normalized_outputs"]
        if output["kind"] in {"proxy_results", "main_results"}
    )
    payload = store.read_json(
        target["contract_ref"],
        schema_file=_schema_file(target["schema_version"]),
    )
    candidate = next(row for row in payload["rows"] if row["role"] == "candidate")
    candidate["metric_value"] = float(candidate["metric_value"]) + 19.0
    reference = store.put_json(payload, schema_file=_schema_file(target["schema_version"]))
    target["contract_ref"] = deepcopy(reference)
    target["content_hash"] = reference["digest"]
    _replace_receipt_output(receipt, target)


def mutate_second_manifest(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    del receipt, store
    manifest["derivation_id"] = "derive:second-hash-consistent-manifest"


def mutate_extra_readiness_source(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    del receipt, store
    source = deepcopy(
        next(item for item in manifest["source_commands"] if "readiness" in item["authority_roles"])
    )
    manifest["source_commands"] = reordinal([*manifest["source_commands"], source])


def mutate_extra_readiness_check(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    target = next(
        output for output in manifest["normalized_outputs"] if output["kind"] == "full_s3_readiness"
    )
    payload = store.read_json(
        target["contract_ref"],
        schema_file=_schema_file(target["schema_version"]),
    )
    extra = deepcopy(payload["checks"][0])
    extra["check_id"] = "readiness-extra-unregistered-check"
    payload["checks"].append(extra)
    reference = store.put_json(payload, schema_file=_schema_file(target["schema_version"]))
    target["contract_ref"] = deepcopy(reference)
    target["content_hash"] = reference["digest"]
    _replace_receipt_output(receipt, target)


def mutate_activation_from_nonactivation_authority(
    manifest: dict[str, Any], receipt: dict[str, Any], store: ContractStore
) -> None:
    del receipt, store
    source = next(
        item for item in manifest["source_commands"] if "activation_surface" in item["authority_roles"]
    )
    source["authority_roles"] = ["normalized_evidence_source"]
    source["readiness_check_ids"] = []


def completed_evidence_manifest(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    state = ResearchEventLedger(root).state()
    assert len(state["trial_results"]) == 0
    proxy_events = [
        event for event in ResearchEventLedger(root).events() if event["event_type"] == "ProxyEvidenceCommitted"
    ]
    assert len(proxy_events) == 1, proxy_events
    attempt_id = proxy_events[0]["payload"]["proxy_outcome"]["attempt_id"]
    return (
        deepcopy(state),
        deepcopy(state["attempts"][attempt_id]),
        deepcopy(proxy_events[0]["payload"]["evidence_manifest"]),
    )


def build_mismatched_evidence_manifest(root: Path) -> dict[str, Any]:
    state, attempt, manifest = completed_evidence_manifest(root)
    validate_receipt_bound_evidence(
        project_root=root,
        attempt=attempt,
        trial_spec=attempt["frozen_trial_spec"],
        manifest=manifest,
        phase_commands=state["phase_commands"],
        phase="proxy",
    )
    store = ContractStore(root)
    original_ref = manifest["derivation_ref"]
    original = store.read_json(
        original_ref,
        schema_file="evidence_derivation_manifest_v3.schema.json",
    )
    second = deepcopy(original)
    second["derivation_id"] = "derive:evidence-manifest-mismatch"
    second_ref = store.put_json(
        second,
        schema_file="evidence_derivation_manifest_v3.schema.json",
    )
    manifest["derivation_ref"] = deepcopy(second_ref)
    manifest["derivation_hash"] = second_ref["digest"]
    for entry in manifest["entries"]:
        entry["derivation_ref"] = deepcopy(second_ref)
        entry["derivation_hash"] = second_ref["digest"]
    validate_schema(manifest, "evidence_manifest_v6.schema.json")
    return manifest


def assert_completed_manifest_rejected_without_writes(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    expected_tokens: tuple[str, ...],
) -> str:
    state, attempt, _ = completed_evidence_manifest(root)
    before = authority_snapshot(root)
    with pytest.raises(ValueError) as captured:
        validate_receipt_bound_evidence(
            project_root=root,
            attempt=attempt,
            trial_spec=attempt["frozen_trial_spec"],
            manifest=manifest,
            phase_commands=state["phase_commands"],
            phase="proxy",
        )
    assert authority_snapshot(root) == before
    _assert_reason(captured.value, expected_tokens)
    return str(captured.value)


def assert_candidate_rejected_without_writes(
    root: Path,
    candidate: DerivationCandidate,
    *,
    expected_tokens: tuple[str, ...],
) -> str:
    ledger = ResearchEventLedger(root)
    derive_record = single_derive_record(ledger.state(), status="started")
    before = authority_snapshot(root)
    with pytest.raises(IntegrityError) as captured:
        ledger.complete_phase_command(
            derive_record["command"]["command_id"],
            candidate.receipt_ref,
        )
    assert authority_snapshot(root) == before
    _assert_reason(captured.value, expected_tokens)
    return str(captured.value)


def authority_snapshot(root: Path) -> dict[str, Any]:
    return {
        "events": _event_rows(root),
        "cas": _tree_snapshot(root, ("meta/contracts/sha256",)),
        "receipt_locators": _tree_snapshot(root, ("meta/command_receipts",)),
        "evidence": _tree_snapshot(root, ("experiment/attempts",)),
        "projections": _tree_snapshot(
            root,
            (
                "meta/research_state.json",
                "meta/attempts",
                "meta/route_outcome.json",
                "meta/direction_outcome_aggregate.json",
                "experiment/results/trial_result.json",
                "plan/trial_spec.json",
                "plan/attempts",
            ),
        ),
        "markers": _tree_snapshot(root, ("runner",)),
    }


def _bind_receipt_to_manifest(
    receipt: dict[str, Any],
    manifest: Mapping[str, Any],
    manifest_ref: Mapping[str, Any],
) -> None:
    receipt["derivation_ref"] = deepcopy(dict(manifest_ref))
    receipt["derivation_hash"] = manifest_ref["digest"]
    for normalized in manifest["normalized_outputs"]:
        _replace_receipt_output(receipt, normalized)


def _replace_receipt_output(receipt: dict[str, Any], normalized: Mapping[str, Any]) -> None:
    matches = [
        output for output in receipt["outputs"] if output["output_id"] == normalized["output_id"]
    ]
    assert len(matches) == 1, (normalized, receipt["outputs"])
    matches[0]["contract_ref"] = deepcopy(normalized["contract_ref"])
    matches[0]["content_hash"] = normalized["content_hash"]


def _schema_file(schema_version: str) -> str:
    prefix = "auto_research_"
    assert schema_version.startswith(prefix)
    return f"{schema_version[len(prefix):]}.schema.json"


def _assert_reason(error: BaseException, expected_tokens: tuple[str, ...]) -> None:
    message = str(error).lower()
    assert all(token.lower() in message for token in expected_tokens), (
        f"wrong validator reason; expected_tokens={expected_tokens} error={error!r}"
    )


def _event_rows(root: Path) -> tuple[tuple[Any, ...], ...]:
    database = root / "meta" / "research_events.sqlite3"
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        return tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT sequence, event_id, event_type, payload_json, previous_event_hash, "
                "event_hash, created_at, schema_version FROM events ORDER BY sequence"
            )
        )
    finally:
        connection.close()


def _tree_snapshot(root: Path, roots: tuple[str, ...]) -> tuple[tuple[str, str, int], ...]:
    records: list[tuple[str, str, int]] = []
    for relative_root in roots:
        path = root / relative_root
        if path.is_symlink():
            records.append((relative_root, f"symlink:{os.readlink(path)}", 0))
            continue
        candidates = [path] if path.is_file() else sorted(path.rglob("*")) if path.is_dir() else []
        for candidate in candidates:
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink():
                records.append((relative, f"symlink:{os.readlink(candidate)}", 0))
            elif candidate.is_file():
                raw = candidate.read_bytes()
                records.append((relative, hashlib.sha256(raw).hexdigest(), len(raw)))
    return tuple(records)


__all__ = [
    "DerivationAttackBaseline",
    "agent_context",
    "assert_candidate_rejected_without_writes",
    "assert_completed_manifest_rejected_without_writes",
    "build_derivation_attack_baseline",
    "build_manifest_candidate",
    "build_mismatched_evidence_manifest",
    "clone_project",
    "mutate_activation_from_nonactivation_authority",
    "mutate_cross_attempt",
    "mutate_cross_generation",
    "mutate_cross_phase",
    "mutate_cross_producer",
    "mutate_duplicate_source",
    "mutate_extra_readiness_check",
    "mutate_extra_readiness_source",
    "mutate_extra_source",
    "mutate_missing_source",
    "mutate_reordered_source",
    "mutate_second_manifest",
    "mutate_self_source",
    "mutate_wrong_decoder_artifact",
    "mutate_wrong_normalized_bytes",
    "mutate_wrong_source_command_id",
    "mutate_wrong_source_output_id",
]
