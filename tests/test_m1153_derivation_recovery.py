from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from auto_research.agents.experiment import ExperimentAgent
from auto_research.contract_store import ContractStore
from auto_research.research_state import IntegrityError, ResearchEventLedger
from auto_research.validators import run_stage_gate
from support.m1152_local_subprocess import c2c_invocations, create_c2c_project, direction_budget


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DERIVE_MARKER = Path("runner") / "m1153_core_derivations.jsonl"


@pytest.fixture(scope="module")
def completed_c2c_source(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    base = tmp_path_factory.mktemp("m1153-derivation-source")
    monkeypatch = pytest.MonkeyPatch()
    try:
        root, _, context, _ = create_c2c_project(
            base,
            monkeypatch,
            profile="standard",
            proxy_accuracy=0.49,
            name="completed-proxy-reject",
        )
        result = ExperimentAgent(context).run()
        assert result["route_outcome"]["next_action"] == "PROPOSE_NEXT_VARIANT"
        gate = run_stage_gate("S3_experiment", root, context.config).to_dict()
        assert gate["status"] == "PASS", gate
        return root, deepcopy(context.config)
    finally:
        monkeypatch.undo()


@pytest.fixture
def completed_c2c_project(
    completed_c2c_source: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> tuple[Path, dict[str, Any]]:
    source, config = completed_c2c_source
    root = tmp_path / source.name
    shutil.copytree(source, root, symlinks=True)
    return root, deepcopy(config)


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


def _file_tree_snapshot(root: Path, relative_roots: tuple[str, ...]) -> tuple[tuple[str, str, int], ...]:
    records: list[tuple[str, str, int]] = []
    for relative_root in relative_roots:
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


def _authority_snapshot(root: Path) -> dict[str, Any]:
    return {
        "events": _event_rows(root),
        "cas": _file_tree_snapshot(root, ("meta/contracts/sha256",)),
        "receipt_locators": _file_tree_snapshot(root, ("meta/command_receipts",)),
        "evidence": _file_tree_snapshot(root, ("experiment/attempts",)),
        "projections": _file_tree_snapshot(
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
    }


def _proxy_event(root: Path) -> dict[str, Any]:
    events = ResearchEventLedger(root).events()
    matches = [event for event in events if event["event_type"] == "ProxyEvidenceCommitted"]
    assert len(matches) == 1
    return matches[0]


def _derive_command_record(root: Path, *, phase: str = "proxy") -> tuple[dict[str, Any], dict[str, Any]]:
    state = ResearchEventLedger(root).state()
    store = ContractStore(root)
    matches = [
        record
        for record in state["phase_commands"].values()
        if record["status"] == "completed"
        and record["command"]["phase"] == phase
        and record["command"]["command_spec_id"].endswith("derive-evidence")
    ]
    assert len(matches) == 1
    record = matches[0]
    receipt = store.read_json(record["receipt_ref"], schema_file="phase_run_receipt_v5.schema.json")
    return record, receipt


def _physical_derivation(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _, receipt = _derive_command_record(root)
    store = ContractStore(root)
    reference = receipt["derivation_ref"]
    assert isinstance(reference, dict)
    assert receipt["derivation_hash"] == reference["digest"]
    manifest = store.read_json(reference, schema_file="evidence_derivation_manifest_v3.schema.json")
    return reference, manifest


def _event_derivation_refs(root: Path) -> list[dict[str, Any]]:
    return [
        entry["derivation_ref"]
        for entry in _proxy_event(root)["payload"]["evidence_manifest"]["entries"]
    ]


def _artifact_reference(root: Path, artifact_kind: str) -> dict[str, Any]:
    store = ContractStore(root)
    event = _proxy_event(root)
    entry = event["payload"]["evidence_manifest"]["entries"][0]
    if artifact_kind == "event_derivation":
        return entry["derivation_ref"]
    if artifact_kind == "normalized":
        return entry["output_ref"]
    physical_ref, physical = _physical_derivation(root)
    if artifact_kind == "physical_derivation":
        return physical_ref
    if artifact_kind == "decoder":
        return physical["decoder_descriptor"]["immutable_ref"]
    if artifact_kind == "raw":
        state = ResearchEventLedger(root).state()
        for record in state["phase_commands"].values():
            command = record.get("command") or {}
            if (
                record.get("status") == "completed"
                and command.get("phase") == "proxy"
                and not str(command.get("command_spec_id") or "").endswith("derive-evidence")
            ):
                receipt = store.read_json(
                    record["receipt_ref"],
                    schema_file="phase_run_receipt_v5.schema.json",
                )
                if receipt.get("raw_outputs"):
                    return receipt["raw_outputs"][0]["contract_ref"]
        raise AssertionError("completed C2C fixture has no physical raw output")
    raise AssertionError(f"unknown artifact kind: {artifact_kind}")


def _mutate_contract(root: Path, reference: dict[str, Any], mutation: str) -> None:
    path = root / reference["relative_path"]
    assert path.is_file()
    if mutation == "delete":
        path.unlink()
    elif mutation == "corrupt":
        path.write_bytes(b'{"tampered":true}')
    else:
        raise AssertionError(f"unknown mutation: {mutation}")


def _invoke_read_validator(
    operation: str,
    root: Path,
    config: dict[str, Any],
    event_id: str,
) -> tuple[BaseException | None, dict[str, Any] | None]:
    error: BaseException | None = None
    report: dict[str, Any] | None = None
    try:
        ledger = ResearchEventLedger(root)
        if operation == "state":
            ledger.state()
        elif operation == "rebuild":
            ledger.rebuild()
        elif operation == "query":
            ledger.query_operation_result(event_id)
        elif operation == "gate":
            report = run_stage_gate("S3_experiment", root, config).to_dict()
        else:
            raise AssertionError(f"unknown operation: {operation}")
    except BaseException as exc:  # The assertion below verifies the exact fail-closed type.
        error = exc
    return error, report


def test_final_evidence_manifest_reuses_the_physical_derivation_reference(
    completed_c2c_project: tuple[Path, dict[str, Any]],
) -> None:
    root, _ = completed_c2c_project
    physical_ref, _ = _physical_derivation(root)
    event_refs = _event_derivation_refs(root)

    assert event_refs
    assert {reference["digest"] for reference in event_refs} == {physical_ref["digest"]}


def test_final_derivation_sources_are_the_frozen_physical_command_exact_set(
    completed_c2c_project: tuple[Path, dict[str, Any]],
) -> None:
    root, _ = completed_c2c_project
    store = ContractStore(root)
    derive_record, _ = _derive_command_record(root)
    _, physical = _physical_derivation(root)
    expected_sources = physical["source_commands"]

    for reference in _event_derivation_refs(root):
        committed = store.read_json(reference, schema_file="evidence_derivation_manifest_v3.schema.json")
        assert committed["source_commands"] == expected_sources
        assert all(
            source["command_id"] != derive_record["command"]["command_id"]
            for source in committed["source_commands"]
        )


@pytest.mark.parametrize("operation", ["state", "rebuild", "query", "gate"])
def test_validator_reads_fail_closed_without_recreating_event_derivation(
    completed_c2c_source: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    operation: str,
) -> None:
    source, config = completed_c2c_source
    root = tmp_path / source.name
    shutil.copytree(source, root, symlinks=True)
    event = _proxy_event(root)
    reference = event["payload"]["evidence_manifest"]["entries"][0]["derivation_ref"]
    _mutate_contract(root, reference, "delete")
    before = _authority_snapshot(root)

    error, report = _invoke_read_validator(operation, root, config, event["event_id"])
    after = _authority_snapshot(root)

    if operation == "gate":
        assert report is not None and report["status"] != "PASS", report
    else:
        assert isinstance(error, IntegrityError), repr(error)
    assert after == before


@pytest.mark.parametrize(
    "artifact_kind",
    ["raw", "decoder", "physical_derivation", "event_derivation", "normalized"],
)
@pytest.mark.parametrize("mutation", ["delete", "corrupt"])
def test_rebuild_fails_closed_without_repairing_immutable_derivation_inputs(
    completed_c2c_source: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    artifact_kind: str,
    mutation: str,
) -> None:
    source, _ = completed_c2c_source
    root = tmp_path / source.name
    shutil.copytree(source, root, symlinks=True)
    event_id = _proxy_event(root)["event_id"]
    reference = _artifact_reference(root, artifact_kind)
    _mutate_contract(root, reference, mutation)
    before = _authority_snapshot(root)

    error, _ = _invoke_read_validator("rebuild", root, {}, event_id)
    after = _authority_snapshot(root)

    assert isinstance(error, IntegrityError), repr(error)
    assert after == before


_COLD_RESTART_SCRIPT = r"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["M1153_TESTS"])

from auto_research.agents.experiment import ExperimentAgent
from auto_research.command_journal import LedgerCommandJournal
from auto_research.contract_store import ContractStore
from auto_research.research_state import ResearchEventLedger
from support.local_c2c_execution import build_c2c_context

root = Path(os.environ["M1153_ROOT"])
repo = Path(os.environ["M1153_REPO"])
marker = root / "runner" / "m1153_core_derivations.jsonl"
crash = os.environ.get("M1153_CRASH", "")

original_put_json = ContractStore.put_json

def tracked_put_json(self, payload, *args, **kwargs):
    reference = original_put_json(self, payload, *args, **kwargs)
    decoder = payload.get("decoder_descriptor") if isinstance(payload, dict) else None
    if (
        isinstance(payload, dict)
        and payload.get("schema_version") == "auto_research_evidence_derivation_manifest_v3"
        and isinstance(decoder, dict)
        and decoder.get("decoder_id") == "c2c-receipt-measurements"
    ):
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "derivation_id": payload["derivation_id"],
                "phase": payload["phase"],
                "digest": reference["digest"],
            }, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return reference

ContractStore.put_json = tracked_put_json

if crash == "receipt-before-completed":
    original_complete = ResearchEventLedger.complete_phase_command
    crashed = {"value": False}

    def crash_complete(self, command_id, receipt_ref):
        record = self.phase_command(command_id) or {}
        command = record.get("command") or {}
        if command.get("command_spec_id", "").endswith("derive-evidence") and not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("m1153 crash after durable derive receipt before Completed")
        return original_complete(self, command_id, receipt_ref)

    ResearchEventLedger.complete_phase_command = crash_complete
elif crash == "completed-before-evidence":
    original_commit_proxy = ResearchEventLedger.commit_proxy_evidence
    crashed = {"value": False}

    def crash_commit_proxy(self, completion_evidence, *, event_id=None):
        if not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("m1153 crash after derive Completed before evidence event")
        return original_commit_proxy(self, completion_evidence, event_id=event_id)

    ResearchEventLedger.commit_proxy_evidence = crash_commit_proxy
elif crash == "orphan-before-locator":
    original_locator = LedgerCommandJournal._write_receipt_locator
    crashed = {"value": False}

    def crash_locator(self, command, receipt_ref):
        if command.get("command_spec_id", "").endswith("derive-evidence") and not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("m1153 crash after orphan derive blobs before locator")
        return original_locator(self, command, receipt_ref)

    LedgerCommandJournal._write_receipt_locator = crash_locator

context = build_c2c_context(root, repo, profile="standard")
result = ExperimentAgent(context).run()
print(json.dumps({
    "route": (result.get("route_outcome") or {}).get("next_action"),
    "state_route": (ResearchEventLedger(root).state().get("last_route_outcome") or {}).get("next_action"),
}, sort_keys=True))
"""


def _run_cold_agent(root: Path, repo: Path, *, crash: str = "") -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "M1153_ROOT": str(root),
            "M1153_REPO": str(repo),
            "M1153_TESTS": str(_PROJECT_ROOT / "tests"),
            "M1153_CRASH": crash,
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(_PROJECT_ROOT / "tests"), environment.get("PYTHONPATH", "")))
            ),
        }
    )
    return subprocess.run(
        [sys.executable, "-c", _COLD_RESTART_SCRIPT],
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def _derive_markers(root: Path) -> list[dict[str, Any]]:
    path = root / _DERIVE_MARKER
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _invocation_counter(root: Path) -> Counter[str]:
    return Counter(json.dumps(record, sort_keys=True) for record in c2c_invocations(root))


def _assert_completed_recovery_once(root: Path, before_invocations: Counter[str]) -> None:
    ledger = ResearchEventLedger(root)
    events = ledger.events()
    event_types = [event["event_type"] for event in events]
    after_invocations = _invocation_counter(root)
    state = ledger.state()

    assert after_invocations == before_invocations
    assert before_invocations and all(count == 1 for count in before_invocations.values())
    assert len(_derive_markers(root)) == 1
    assert event_types.count("ProxyEvidenceCommitted") == 1
    assert event_types.count("PhaseCommandCompleted") == len(
        [record for record in state["phase_commands"].values() if record["status"] == "completed"]
    )
    assert state["last_route_outcome"]["next_action"] == "PROPOSE_NEXT_VARIANT"
    assert direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 0}
    physical_ref, _ = _physical_derivation(root)
    assert {reference["digest"] for reference in _event_derivation_refs(root)} == {physical_ref["digest"]}


def test_cold_restart_reconciles_durable_derive_receipt_before_completed_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, repo, _, _ = create_c2c_project(
        tmp_path,
        monkeypatch,
        profile="standard",
        proxy_accuracy=0.49,
        name="cold-receipt-before-completed",
    )
    crashed = _run_cold_agent(root, repo, crash="receipt-before-completed")
    assert crashed.returncode != 0
    assert "after durable derive receipt before Completed" in crashed.stderr
    before_invocations = _invocation_counter(root)
    assert len(_derive_markers(root)) == 1
    state = ResearchEventLedger(root).state()
    derive_records = [
        record
        for record in state["phase_commands"].values()
        if record["command"]["command_spec_id"].endswith("derive-evidence")
    ]
    assert len(derive_records) == 1 and derive_records[0]["status"] == "started"

    restarted = _run_cold_agent(root, repo)
    assert restarted.returncode == 0, restarted.stderr
    _assert_completed_recovery_once(root, before_invocations)


def test_cold_restart_reuses_completed_derive_before_evidence_event_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, repo, _, _ = create_c2c_project(
        tmp_path,
        monkeypatch,
        profile="standard",
        proxy_accuracy=0.49,
        name="cold-completed-before-evidence",
    )
    crashed = _run_cold_agent(root, repo, crash="completed-before-evidence")
    assert crashed.returncode != 0
    assert "after derive Completed before evidence event" in crashed.stderr
    before_invocations = _invocation_counter(root)
    assert len(_derive_markers(root)) == 1
    state = ResearchEventLedger(root).state()
    assert any(
        record["status"] == "completed"
        and record["command"]["command_spec_id"].endswith("derive-evidence")
        for record in state["phase_commands"].values()
    )
    assert "ProxyEvidenceCommitted" not in [event["event_type"] for event in ResearchEventLedger(root).events()]

    restarted = _run_cold_agent(root, repo)
    assert restarted.returncode == 0, restarted.stderr
    _assert_completed_recovery_once(root, before_invocations)


def test_orphan_derive_blobs_have_no_recovery_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, repo, _, _ = create_c2c_project(
        tmp_path,
        monkeypatch,
        profile="standard",
        proxy_accuracy=0.49,
        name="cold-orphan-before-locator",
    )
    crashed = _run_cold_agent(root, repo, crash="orphan-before-locator")
    assert crashed.returncode != 0
    assert "after orphan derive blobs before locator" in crashed.stderr
    before_invocations = _invocation_counter(root)
    markers = _derive_markers(root)
    assert len(markers) == 1
    orphan_digest = markers[0]["digest"]
    assert (root / "meta" / "contracts" / "sha256" / orphan_digest[:2] / f"{orphan_digest}.json").is_file()

    restarted = _run_cold_agent(root, repo)
    assert restarted.returncode == 0, restarted.stderr
    ledger = ResearchEventLedger(root)
    events = ledger.events()
    state = ledger.state()

    assert _invocation_counter(root) == before_invocations
    assert before_invocations and all(count == 1 for count in before_invocations.values())
    assert len(_derive_markers(root)) == 1
    assert state["last_route_outcome"]["next_action"] == "BLOCK_INTEGRITY"
    assert direction_budget(root) == {"target": 5, "reserved": 0, "consumed": 0}
    assert "ProxyEvidenceCommitted" not in [event["event_type"] for event in events]
    assert orphan_digest not in json.dumps(events, sort_keys=True)
