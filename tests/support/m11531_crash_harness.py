from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from auto_research.contract_store import ContractStore
from auto_research.research_state import ResearchEventLedger
from support.m1152_local_subprocess import (
    c2c_invocations,
    create_c2c_project,
    direction_budget,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCING_DERIVE_MARKER = Path("runner") / "m11531_producing_derivations.jsonl"
VALIDATOR_MARKER = Path("runner") / "m11531_validator_recomputations.jsonl"


CRASH_AFTER_PHYSICAL_BEFORE_DERIVE_STARTED = "physical-before-derive-started"
CRASH_AFTER_DERIVED_BLOBS_BEFORE_LOCATOR = "derived-blobs-before-locator"
CRASH_AFTER_DERIVE_RECEIPT_BEFORE_COMPLETED = "derive-receipt-before-completed"
CRASH_AFTER_DERIVE_COMPLETED_BEFORE_EVIDENCE = "derive-completed-before-evidence"
CRASH_AFTER_READINESS_RECEIPT_BEFORE_COMPLETED = "readiness-receipt-before-completed"
CRASH_AFTER_READINESS_COMPLETED_BEFORE_PROXY_COMMIT = "readiness-completed-before-proxy-commit"
CRASH_AFTER_PROXY_COMMIT_BEFORE_ROUTE = "proxy-commit-before-route"


@dataclass(frozen=True)
class ColdRun:
    returncode: int
    stdout: str
    stderr: str
    result: dict[str, Any] | None


@dataclass(frozen=True)
class CrashObservation:
    last_sequence: int
    event_types: tuple[str, ...]
    physical_invocations: int
    physical_unique_invocations: int
    physical_max_repeat: int
    producing_derive_invocations: int
    producing_derive_by_phase: dict[str, int]
    validator_recomputations: int
    validator_by_phase: dict[str, int]
    derive_receipt_hash: str | None
    derivation_manifest_hash: str | None
    orphan_derivation_manifest_hashes: tuple[str, ...]
    last_route: str | None
    last_route_source_event_id: str | None
    proxy_decisions: tuple[str, ...]
    budget: dict[str, int]
    full_phase_started: int
    proxy_evidence_committed: int
    attempt_finalized: int
    phase_command_started: int
    phase_command_completed: int
    phase_command_unknown: int

    def diagnostic(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)


def create_crash_project(
    tmp_path: Path,
    monkeypatch: Any,
    *,
    name: str,
    proxy_accuracy: float,
    readiness_blocked: bool = False,
) -> tuple[Path, Path]:
    root, repo, _, _ = create_c2c_project(
        tmp_path,
        monkeypatch,
        profile="standard",
        proxy_accuracy=proxy_accuracy,
        name=name,
    )
    if readiness_blocked:
        control_path = repo / "local_execution_control.json"
        control = json.loads(control_path.read_text(encoding="utf-8"))
        control["readiness_checks"]["proxy-ready-for-full"]["measurement"] = 0.0
        control_path.write_text(json.dumps(control, sort_keys=True), encoding="utf-8")
    return root, repo


def run_cold_agent(root: Path, repo: Path, *, crash_point: str = "") -> ColdRun:
    environment = os.environ.copy()
    environment.update(
        {
            "M11531_ROOT": str(root),
            "M11531_REPO": str(repo),
            "M11531_TESTS": str(PROJECT_ROOT / "tests"),
            "M11531_CRASH_POINT": crash_point,
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(PROJECT_ROOT / "tests"), environment.get("PYTHONPATH", "")))
            ),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", _COLD_AGENT_SCRIPT],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    result = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("m11531_cold_result") is True:
            result = candidate
            break
    return ColdRun(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        result=result,
    )


def observe_authority(root: Path) -> CrashObservation:
    ledger = ResearchEventLedger(root)
    events = ledger.events()
    event_types = tuple(str(event["event_type"]) for event in events)
    state = ledger.state()
    physical_records = c2c_invocations(root)
    physical_counter = Counter(_canonical_record(record) for record in physical_records)
    producing = _marker_records(root / PRODUCING_DERIVE_MARKER)
    validators = _marker_records(root / VALIDATOR_MARKER)
    derive_receipt_hash, derivation_hash = _derive_authority_hashes(root, state)
    route = state.get("last_route_outcome")
    proxy_decisions = tuple(
        str(event["payload"]["proxy_outcome"]["decision"])
        for event in events
        if event["event_type"] == "ProxyEvidenceCommitted"
    )
    return CrashObservation(
        last_sequence=int(state["last_sequence"]),
        event_types=event_types,
        physical_invocations=len(physical_records),
        physical_unique_invocations=len(physical_counter),
        physical_max_repeat=max(physical_counter.values(), default=0),
        producing_derive_invocations=len(producing),
        producing_derive_by_phase=dict(Counter(str(item.get("phase")) for item in producing)),
        validator_recomputations=len(validators),
        validator_by_phase=dict(Counter(str(item.get("phase")) for item in validators)),
        derive_receipt_hash=derive_receipt_hash,
        derivation_manifest_hash=derivation_hash,
        orphan_derivation_manifest_hashes=_derivation_manifest_hashes(root),
        last_route=(str(route.get("next_action")) if isinstance(route, Mapping) else None),
        last_route_source_event_id=(
            str((route.get("source") or {}).get("event_id"))
            if isinstance(route, Mapping) and (route.get("source") or {}).get("event_id")
            else None
        ),
        proxy_decisions=proxy_decisions,
        budget=direction_budget(root),
        full_phase_started=event_types.count("FullPhaseStarted"),
        proxy_evidence_committed=event_types.count("ProxyEvidenceCommitted"),
        attempt_finalized=event_types.count("AttemptFinalized"),
        phase_command_started=event_types.count("PhaseCommandStarted"),
        phase_command_completed=event_types.count("PhaseCommandCompleted"),
        phase_command_unknown=event_types.count("PhaseCommandUnknownOutcome"),
    )


def raw_event_rows(root: Path) -> tuple[tuple[Any, ...], ...]:
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


def assert_legal_baseline(observation: CrashObservation, cold_run: ColdRun) -> None:
    assert cold_run.returncode == 0, cold_run.stderr
    assert cold_run.result is not None, cold_run.stdout
    assert cold_run.result["result_route"] == "PROPOSE_NEXT_VARIANT", cold_run.result
    assert observation.last_route == "PROPOSE_NEXT_VARIANT", observation.diagnostic()
    assert observation.proxy_decisions == ("PROPOSE_NEXT_VARIANT",), observation.diagnostic()
    assert observation.physical_invocations > 0, observation.diagnostic()
    assert observation.physical_max_repeat == 1, observation.diagnostic()
    assert observation.producing_derive_by_phase == {"proxy": 1}, observation.diagnostic()
    assert observation.validator_recomputations > 0, observation.diagnostic()
    assert observation.proxy_evidence_committed == 1, observation.diagnostic()
    assert observation.full_phase_started == 0, observation.diagnostic()
    assert observation.attempt_finalized == 0, observation.diagnostic()
    assert observation.budget == {"target": 5, "reserved": 0, "consumed": 0}, observation.diagnostic()


def assert_injected_crash(cold_run: ColdRun, expected_fragment: str) -> None:
    assert cold_run.returncode != 0, cold_run.stdout
    assert expected_fragment in cold_run.stderr, cold_run.stderr


def assert_physical_not_replayed(before: CrashObservation, after: CrashObservation) -> None:
    assert after.physical_invocations == before.physical_invocations, after.diagnostic()
    assert after.physical_unique_invocations == before.physical_unique_invocations, after.diagnostic()
    assert after.physical_max_repeat == 1, after.diagnostic()


def assert_event_chain_prefix(before: tuple[tuple[Any, ...], ...], after: tuple[tuple[Any, ...], ...]) -> None:
    assert after[: len(before)] == before
    assert [int(row[0]) for row in after] == list(range(1, len(after) + 1))


def _canonical_record(record: Mapping[str, Any]) -> str:
    return json.dumps(dict(record), sort_keys=True, separators=(",", ":"))


def _marker_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _derive_authority_hashes(
    root: Path,
    state: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    records = [
        record
        for record in state.get("phase_commands", {}).values()
        if isinstance(record, Mapping)
        and str((record.get("command") or {}).get("command_spec_id", "")).endswith("derive-evidence")
        and (record.get("command") or {}).get("phase") == "proxy"
    ]
    if not records:
        return None, None
    assert len(records) == 1
    record = records[0]
    receipt_ref = record.get("receipt_ref")
    if not isinstance(receipt_ref, Mapping):
        command_id = str((record.get("command") or {}).get("command_id") or "")
        receipt_ref = _locator_receipt_ref(root, command_id)
    if not isinstance(receipt_ref, Mapping):
        return None, None
    receipt_hash = str(receipt_ref.get("digest") or "") or None
    if receipt_hash is None:
        return None, None
    try:
        receipt = ContractStore(root).read_json(receipt_ref)
    except (OSError, TypeError, ValueError):
        return receipt_hash, None
    derivation_hash = receipt.get("derivation_hash")
    return receipt_hash, str(derivation_hash) if isinstance(derivation_hash, str) else None


def _locator_receipt_ref(root: Path, command_id: str) -> dict[str, Any] | None:
    locator_root = root / "meta" / "command_receipts"
    if not locator_root.is_dir():
        return None
    for path in sorted(locator_root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("command_id") == command_id and isinstance(payload.get("receipt_ref"), dict):
            return dict(payload["receipt_ref"])
    return None


def _derivation_manifest_hashes(root: Path) -> tuple[str, ...]:
    contract_root = root / "meta" / "contracts" / "sha256"
    hashes: list[str] = []
    if not contract_root.is_dir():
        return ()
    for path in sorted(contract_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if str(payload.get("schema_version") or "").startswith(
            "auto_research_evidence_derivation_manifest_v"
        ):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes.append(digest)
    return tuple(sorted(hashes))


_COLD_AGENT_SCRIPT = r'''
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["M11531_TESTS"])

import auto_research.agents.experiment as experiment_module
import auto_research.derivation_validation as derivation_validation
import auto_research.evidence_lineage as evidence_lineage
from auto_research.agents.experiment import ExperimentAgent
from auto_research.command_journal import LedgerCommandJournal
from auto_research.research_state import ResearchEventLedger
from support.local_c2c_execution import build_c2c_context

root = Path(os.environ["M11531_ROOT"])
repo = Path(os.environ["M11531_REPO"])
crash_point = os.environ.get("M11531_CRASH_POINT", "")
derive_marker = root / "runner" / "m11531_producing_derivations.jsonl"
validator_marker = root / "runner" / "m11531_validator_recomputations.jsonl"


def append_marker(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


original_producing_decoder = experiment_module.derive_evidence_deterministically


def tracked_producing_decoder(*args, **kwargs):
    result = original_producing_decoder(*args, **kwargs)
    append_marker(derive_marker, {
        "attempt_id": str((kwargs.get("attempt") or {}).get("attempt_id") or ""),
        "phase": str(kwargs.get("phase") or ""),
        "plan_id": str((kwargs.get("derivation_plan") or {}).get("plan_id") or ""),
        "normalized_hashes": [item.content_hash for item in result.normalized_outputs],
    })
    return result


experiment_module.derive_evidence_deterministically = tracked_producing_decoder

original_validator = derivation_validation.validate_immutable_derivation


def tracked_validator(*args, **kwargs):
    append_marker(validator_marker, {
        "attempt_id": str((kwargs.get("attempt") or {}).get("attempt_id") or ""),
        "phase": str(kwargs.get("phase") or ""),
        "operation": "validate_immutable_derivation",
    })
    return original_validator(*args, **kwargs)


derivation_validation.validate_immutable_derivation = tracked_validator
evidence_lineage.validate_immutable_derivation = tracked_validator


def command_spec_for(ledger, command_id):
    record = ledger.phase_command(command_id) or {}
    command = record.get("command") or {}
    return str(command.get("command_spec_id") or "")


if crash_point == "physical-before-derive-started":
    original_start = ResearchEventLedger.start_phase_command
    crashed = {"value": False}

    def crash_start(self, command):
        if str(command.get("command_spec_id") or "").endswith("derive-evidence") and not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("m11531 crash after physical receipts before derive PhaseCommandStarted")
        return original_start(self, command)

    ResearchEventLedger.start_phase_command = crash_start
elif crash_point == "derived-blobs-before-locator":
    original_locator = LedgerCommandJournal._write_receipt_locator
    crashed = {"value": False}

    def crash_locator(self, command, receipt_ref):
        if str(command.get("command_spec_id") or "").endswith("derive-evidence") and not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("m11531 crash after normalized blobs and derivation manifest before locator")
        return original_locator(self, command, receipt_ref)

    LedgerCommandJournal._write_receipt_locator = crash_locator
elif crash_point == "derive-receipt-before-completed":
    original_complete = ResearchEventLedger.complete_phase_command
    crashed = {"value": False}

    def crash_complete(self, command_id, receipt_ref):
        if command_spec_for(self, command_id).endswith("derive-evidence") and not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("m11531 crash after durable derive receipt before PhaseCommandCompleted")
        return original_complete(self, command_id, receipt_ref)

    ResearchEventLedger.complete_phase_command = crash_complete
elif crash_point == "derive-completed-before-evidence":
    original_commit_proxy = ResearchEventLedger.commit_proxy_evidence
    crashed = {"value": False}

    def crash_commit_proxy(self, completion_evidence, *, event_id=None):
        if not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("m11531 crash after derive Completed before ProxyEvidenceCommitted")
        return original_commit_proxy(self, completion_evidence, event_id=event_id)

    ResearchEventLedger.commit_proxy_evidence = crash_commit_proxy
elif crash_point == "readiness-receipt-before-completed":
    original_complete = ResearchEventLedger.complete_phase_command
    crashed = {"value": False}

    def crash_readiness_complete(self, command_id, receipt_ref):
        command_spec_id = command_spec_for(self, command_id)
        if "activation_smoke" in command_spec_id and not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("m11531 crash after readiness physical receipt before PhaseCommandCompleted")
        return original_complete(self, command_id, receipt_ref)

    ResearchEventLedger.complete_phase_command = crash_readiness_complete
elif crash_point == "readiness-completed-before-proxy-commit":
    original_derive_phase = ExperimentAgent._commit_phase_evidence_derivation
    crashed = {"value": False}

    def crash_before_proxy_derivation(self):
        if self._active_phase_context.phase == "proxy" and not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("m11531 crash after readiness Completed before proxy evidence commit")
        return original_derive_phase(self)

    ExperimentAgent._commit_phase_evidence_derivation = crash_before_proxy_derivation
elif crash_point == "proxy-commit-before-route":
    original_commit_proxy = ResearchEventLedger.commit_proxy_evidence
    crashed = {"value": False}

    def crash_after_proxy_commit(self, completion_evidence, *, event_id=None):
        committed = original_commit_proxy(self, completion_evidence, event_id=event_id)
        if not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("m11531 crash after ProxyEvidenceCommitted before route delivery")
        return committed

    ResearchEventLedger.commit_proxy_evidence = crash_after_proxy_commit

context = build_c2c_context(root, repo, profile="standard")
result = ExperimentAgent(context).run()
state = ResearchEventLedger(root).state()
route = result.get("route_outcome") or {}
state_route = state.get("last_route_outcome") or {}
print(json.dumps({
    "m11531_cold_result": True,
    "result_route": route.get("next_action"),
    "result_source_event_id": (route.get("source") or {}).get("event_id"),
    "state_route": state_route.get("next_action"),
    "state_route_source_event_id": (state_route.get("source") or {}).get("event_id"),
    "last_sequence": state.get("last_sequence"),
}, sort_keys=True))
'''


__all__ = [
    "CRASH_AFTER_DERIVED_BLOBS_BEFORE_LOCATOR",
    "CRASH_AFTER_DERIVE_COMPLETED_BEFORE_EVIDENCE",
    "CRASH_AFTER_DERIVE_RECEIPT_BEFORE_COMPLETED",
    "CRASH_AFTER_PHYSICAL_BEFORE_DERIVE_STARTED",
    "CRASH_AFTER_PROXY_COMMIT_BEFORE_ROUTE",
    "CRASH_AFTER_READINESS_COMPLETED_BEFORE_PROXY_COMMIT",
    "CRASH_AFTER_READINESS_RECEIPT_BEFORE_COMPLETED",
    "ColdRun",
    "CrashObservation",
    "assert_event_chain_prefix",
    "assert_injected_crash",
    "assert_legal_baseline",
    "assert_physical_not_replayed",
    "create_crash_project",
    "observe_authority",
    "raw_event_rows",
    "run_cold_agent",
]
