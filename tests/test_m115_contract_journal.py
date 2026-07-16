from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import pytest

from auto_research.contract_store import ContractStore, canonical_contract_bytes, contract_digest, validate_schema
from auto_research.phase_command_plan import build_phase_command_plan, store_phase_command_plan


def test_contract_store_digest_proves_exact_persisted_bytes(tmp_path: Path) -> None:
    store = ContractStore(tmp_path)
    payload = {"schema_version": "fixture_v1", "value": "真实 bytes"}
    raw = canonical_contract_bytes(payload)
    reference = store.put_bytes(raw)
    assert reference["digest"] == contract_digest(raw)
    assert store.read_bytes(reference) == raw
    assert store.put_bytes(raw) == reference

    path = tmp_path / reference["relative_path"]
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="size|digest"):
        store.read_bytes(reference)


def test_contract_store_rejects_claimed_digest_and_symlink_path(tmp_path: Path) -> None:
    store = ContractStore(tmp_path)
    raw = b"{}"
    with pytest.raises(ValueError, match="expected digest"):
        store.put_bytes(raw, expected_digest="0" * 64)

    digest = contract_digest(raw)
    reference = store.reference(digest, size_bytes=len(raw))
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "meta").mkdir()
    os.symlink(outside, tmp_path / "meta" / "contracts")
    with pytest.raises(ValueError, match="symlink|unavailable"):
        store.write_bytes(reference, raw)


def test_contract_store_rejects_leaf_symlink_and_hard_link(tmp_path: Path) -> None:
    store = ContractStore(tmp_path)
    raw = b'{"contract":"immutable"}'
    reference = store.put_bytes(raw)
    path = tmp_path / reference["relative_path"]

    hard_link = tmp_path / "hard-link.json"
    os.link(path, hard_link)
    with pytest.raises(ValueError, match="hard links"):
        store.read_bytes(reference)
    hard_link.unlink()

    outside = tmp_path / "outside-contract.json"
    outside.write_bytes(raw)
    path.unlink()
    os.symlink(outside, path)
    with pytest.raises(ValueError, match="symlink|unavailable"):
        store.read_bytes(reference)


def test_sample_v3_and_evaluator_v2_bind_cas_references(tmp_path: Path) -> None:
    store = ContractStore(tmp_path)
    sample_blob = store.put_bytes(b'{"sample":1}')
    evaluator_blob = store.put_bytes(b'{"evaluator":"v2"}')
    evaluator_config = store.put_bytes(b'{"metric":"accuracy"}')
    sample_manifest = {
        "schema_version": "auto_research_sample_manifest_v4",
        "manifest_id": "samples-m115",
        "provenance_mode": "real",
        "datasets": [{
            "dataset_id": "dataset-a",
            "source_revision": "revision-1",
            "split": "test",
            "sample_count": 1,
            "ordered_sample_ids": [sample_blob["digest"]],
            "raw_sample_refs": [sample_blob],
            "content_digest": sample_blob["digest"],
            "record_format": "jsonl-record-bytes-v1",
            "canonicalization_contract": "preserve-selected-record-bytes-v1",
        }],
    }
    evaluator_manifest = {
        "schema_version": "auto_research_evaluator_manifest_v2",
        "evaluator_id": "evaluator-m115",
        "provenance_mode": "real",
        "source_blobs": [evaluator_blob],
        "dependency_blobs": [],
        "config_blob": evaluator_config,
        "config_digest": evaluator_config["digest"],
        "source_digest": evaluator_blob["digest"],
        "dependency_digest": contract_digest(b""),
    }
    validate_schema(sample_manifest, "sample_manifest_v4.schema.json")
    validate_schema(evaluator_manifest, "evaluator_manifest_v2.schema.json")
    assert store.read_json(store.put_json(sample_manifest, schema_file="sample_manifest_v4.schema.json")) == sample_manifest
    assert store.read_json(store.put_json(evaluator_manifest, schema_file="evaluator_manifest_v2.schema.json")) == evaluator_manifest
    sample_ref = store.put_contract(
        sample_manifest,
        contract_kind="sample_manifest",
        schema_file="sample_manifest_v4.schema.json",
    )
    assert store.read_contract(
        sample_ref,
        contract_kind="sample_manifest",
        schema_file="sample_manifest_v4.schema.json",
    ) == sample_manifest
    with pytest.raises(ValueError, match="kind mismatch"):
        store.read_contract(
            sample_ref,
            contract_kind="evaluator_manifest",
            schema_file="sample_manifest_v4.schema.json",
        )

    evaluator_manifest["source_digest"] = "9" * 64
    with pytest.raises(ValueError, match="source_digest"):
        store.put_json(evaluator_manifest, schema_file="evaluator_manifest_v2.schema.json")


from auto_research.command_journal import CommandExecutionResult, CommandJournalError, LedgerCommandJournal
from auto_research.phase_execution import AuthoritativePhaseContext, PhaseAuthorization


class FakeLedger:
    def __init__(self, authorization: PhaseAuthorization):
        self.authorization = authorization
        self.commands: dict[str, dict] = {}
        self.sequence = 10
        self.complete_error = False
        self.pending_receipt = None
        self.authorization_calls = 0

    def authorize_phase(self, context):
        self.authorization_calls += 1
        return self.authorization

    def phase_command(self, command_id):
        return self.commands.get(command_id)

    def start_phase_command(self, command):
        current = self.commands.get(command["command_id"])
        if current is not None:
            return current
        self.sequence += 1
        current = {
            "status": "started", "command": dict(command), "event_id": f"event:command:{command['command_id']}",
            "event_hash": "a" * 64, "created_at": "2026-07-15T00:00:00Z", "sequence": self.sequence,
        }
        self.commands[command["command_id"]] = current
        return current

    def complete_phase_command(self, command_id, receipt_ref):
        if self.complete_error:
            self.pending_receipt = dict(receipt_ref)
            raise RuntimeError("simulated DB crash")
        current = self.commands[command_id]
        current.update(status="completed", receipt=dict(receipt_ref))
        return current

    def mark_phase_command_unknown(self, command_id, reason):
        current = self.commands[command_id]
        current.update(status="unknown", unknown_reason=reason)
        return current


def _command_context(
    tmp_path: Path,
    *,
    commands: tuple[dict, ...] | None = None,
) -> tuple[AuthoritativePhaseContext, PhaseAuthorization]:
    (tmp_path / "work").mkdir(exist_ok=True)
    source_snapshot_hash = "f" * 64
    command_values = commands or ({"command_spec_id": "full-command-train", "argv": ["python", "train.py"]},)
    plan = build_phase_command_plan(
        phase="full",
        adapter_id="adapter-command-1",
        adapter_version="1",
        provenance_mode="local-external",
        variant_spec_hash="c" * 64,
        source_snapshot_hash=source_snapshot_hash,
        command_values=command_values,
        expected_evidence=[{"kind": "main_results", "schema_version": "auto_research_main_results_v3"}],
        default_cwd="work",
    )
    _, command_plan_hash = store_phase_command_plan(tmp_path, plan)
    authorization = PhaseAuthorization(
        attempt_id="attempt-command-1", lifecycle_generation=0, phase="full",
        phase_execution_id="phase-full-command", phase_start_event_id="event:full:start",
        phase_start_event_hash="1" * 64, phase_start_sequence=3, producer_run_id="producer-command-1",
        implementation_hash="2" * 64, attempt_input_hash="3" * 64, trial_spec_hash="4" * 64,
        command_plan_hash=command_plan_hash, phase_contract_hash="6" * 64,
        expected_evidence_kinds=("main_results",), adapter_identity="adapter-command-1",
        provenance_mode="local_external", state="FULL_RUNNING",
        proxy_commit_event_id="event:proxy:commit", proxy_commit_event_hash="7" * 64,
        proxy_outcome_hash="8" * 64,
    )
    attempt = {
        "attempt_id": authorization.attempt_id, "direction_semantic_hash": "9" * 64,
        "direction_spec_hash": "a" * 64, "variant_semantic_hash": "b" * 64,
        "variant_spec_hash": "c" * 64, "trial_spec_hash": authorization.trial_spec_hash,
        "lifecycle_generation": 0, "implementation_hash": authorization.implementation_hash,
        "attempt_input_hash": authorization.attempt_input_hash,
    }
    return AuthoritativePhaseContext.from_authorization(tmp_path, attempt, authorization), authorization


def _command_result(tmp_path: Path) -> CommandExecutionResult:
    output = ContractStore(tmp_path).put_bytes(b'{"metric":1}')
    stdout = "fixture command completed"
    stderr = ""
    return CommandExecutionResult(
        exit_code=0,
        stdout_hash=hashlib.sha256(stdout.encode()).hexdigest(),
        stderr_hash=hashlib.sha256(stderr.encode()).hexdigest(),
        outputs=(output,),
        external_job_id="job-1",
        stdout=stdout,
        stderr=stderr,
    )


def test_ledger_journal_executes_completed_command_exactly_once(tmp_path: Path) -> None:
    context, authorization = _command_context(tmp_path)
    ledger = FakeLedger(authorization)
    journal = LedgerCommandJournal(tmp_path, ledger)
    calls: list[str] = []

    def runner():
        calls.append("run")
        return _command_result(tmp_path)

    kwargs = dict(
        command_id="command-full-0001", argv=("python", "train.py"), cwd="work",
        source_snapshot_hash="f" * 64, expected_outputs=("main_results",), runner=runner,
    )
    first = journal.run_once(context, **kwargs)
    second = journal.run_once(context, **kwargs)
    assert first["status"] == second["status"] == "completed"
    assert calls == ["run"]
    assert ledger.authorization_calls == 3
    receipt = ContractStore(tmp_path).read_json(first["receipt"], schema_file="phase_run_receipt_v4.schema.json")
    assert receipt["phase_start_event_id"] == context.phase_start_event_id
    assert receipt["producer_run_id"] == context.producer_run_id


def test_started_without_receipt_becomes_unknown_and_never_reruns(tmp_path: Path) -> None:
    context, authorization = _command_context(tmp_path)
    ledger = FakeLedger(authorization)
    journal = LedgerCommandJournal(tmp_path, ledger)
    command = journal._command_payload(
        context, authorization, command_id="command-full-0002", argv=("python", "train.py"), cwd="work",
        source_snapshot_hash="f" * 64, expected_outputs=("main_results",),
    )
    ledger.start_phase_command(command)
    calls: list[str] = []
    result = journal.run_once(
        context, command_id="command-full-0002", argv=("python", "train.py"), cwd="work",
        source_snapshot_hash="f" * 64, expected_outputs=("main_results",),
        runner=lambda: calls.append("run") or _command_result(tmp_path),
    )
    assert result["status"] == "unknown"
    assert calls == []


def test_receipt_after_side_effect_before_db_can_be_reconciled_without_rerun(tmp_path: Path) -> None:
    context, authorization = _command_context(tmp_path)
    ledger = FakeLedger(authorization)
    ledger.complete_error = True
    journal = LedgerCommandJournal(tmp_path, ledger)
    calls: list[str] = []
    kwargs = dict(
        command_id="command-full-0003", argv=("python", "train.py"), cwd="work",
        source_snapshot_hash="f" * 64, expected_outputs=("main_results",),
        runner=lambda: calls.append("run") or _command_result(tmp_path),
    )
    with pytest.raises(RuntimeError, match="DB crash"):
        journal.run_once(context, **kwargs)
    ledger.complete_error = False
    recovered = journal.run_once(context, recovered_receipt_ref=ledger.pending_receipt, **kwargs)
    assert recovered["status"] == "completed"
    assert calls == ["run"]


def test_command_id_reuse_with_different_intent_is_integrity_conflict(tmp_path: Path) -> None:
    context, authorization = _command_context(
        tmp_path,
        commands=(
            {
                "command_spec_id": "full-command-true",
                "argv": ["true"],
                "expected_outputs": [{"kind": "main_results", "schema_version": "auto_research_main_results_v3", "required": True}],
            },
            {
                "command_spec_id": "full-command-false",
                "argv": ["false"],
                "expected_outputs": [],
            },
        ),
    )
    ledger = FakeLedger(authorization)
    journal = LedgerCommandJournal(tmp_path, ledger)
    journal.run_once(
        context, command_id="command-full-0004", argv=("true",), cwd="work",
        source_snapshot_hash="f" * 64, expected_outputs=("main_results",), runner=lambda: _command_result(tmp_path),
    )
    with pytest.raises(CommandJournalError, match="conflicts"):
        journal.run_once(
            context, command_id="command-full-0004", argv=("false",), cwd="work",
            source_snapshot_hash="f" * 64, expected_outputs=(), runner=lambda: _command_result(tmp_path),
        )


def test_ledger_journal_requires_exact_frozen_source_and_policies(tmp_path: Path) -> None:
    policies = {
        "retry_policy": {"max_attempts": 2, "retryable_exit_codes": [75], "backoff_seconds": 1},
        "resource_policy": {"resource_class": "cpu", "minimum_capacity": 2, "unit": "count"},
        "resume_policy": {"mode": "receipt_only", "external_job_attach_required": False},
    }
    context, authorization = _command_context(
        tmp_path,
        commands=({"command_spec_id": "full-command-policy", "argv": ["python", "train.py"], **policies},),
    )
    journal = LedgerCommandJournal(tmp_path, FakeLedger(authorization))
    common = {
        "command_id": "command-full-policy",
        "command_spec_id": "full-command-policy",
        "argv": ("python", "train.py"),
        "cwd": "work",
        "expected_outputs": ("main_results",),
        "runner": lambda: _command_result(tmp_path),
    }
    with pytest.raises(CommandJournalError, match="source snapshot"):
        journal.run_once(context, source_snapshot_hash=context.implementation_hash, **common)
    with pytest.raises(CommandJournalError, match="retry_policy"):
        journal.run_once(
            context,
            source_snapshot_hash="f" * 64,
            retry_policy={"max_attempts": 1, "retryable_exit_codes": [], "backoff_seconds": 0},
            **common,
        )
    assert journal.run_once(context, source_snapshot_hash="f" * 64, **policies, **common)["status"] == "completed"


def test_ledger_journal_rejects_symlinked_cwd_before_start(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "linked-work").symlink_to(outside, target_is_directory=True)
    context, authorization = _command_context(
        tmp_path,
        commands=({"command_spec_id": "full-command-linked", "argv": ["python", "train.py"], "cwd": "linked-work"},),
    )
    ledger = FakeLedger(authorization)
    with pytest.raises(CommandJournalError, match="symlink"):
        LedgerCommandJournal(tmp_path, ledger).run_once(
            context,
            command_id="command-full-linked",
            command_spec_id="full-command-linked",
            argv=("python", "train.py"),
            cwd="linked-work",
            source_snapshot_hash="f" * 64,
            expected_outputs=("main_results",),
            runner=lambda: _command_result(tmp_path),
        )
    assert ledger.commands == {}
