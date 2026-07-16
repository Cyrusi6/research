"""SQLite-WAL authoritative event store and deterministic S1-S3 reducer."""

from __future__ import annotations

import json
import hashlib
import os
import re
import sqlite3
import tempfile
import threading
import time
import uuid
import fcntl
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from .domain_contracts import (
    ATTEMPT_SCHEMA_VERSION,
    DIRECTION_AGGREGATE_SCHEMA_VERSION,
    EVENT_SCHEMA_VERSION,
    FAILURE_EVIDENCE_SCHEMA_VERSION,
    PHASE_EXECUTION_MANIFEST_SCHEMA_VERSION,
    RESEARCH_STATE_SCHEMA_VERSION,
    RESUME_EVIDENCE_SCHEMA_VERSION,
    acceptance_contract_hash,
    attempt_input_hash,
    canonical_hash,
    classify_trial_result,
    trial_spec_hash,
    validate_contract,
    validate_direction_identity,
    validate_trial_spec,
    validate_trial_evidence,
    validate_trial_result,
    validate_variant_identity,
)
from .s3_validation import validate_ledger_trial_precommit
from .contract_store import ContractStore, canonical_contract_bytes, contract_digest
from .evidence import EvidenceStore, content_addressed_evidence_path
from .evidence_lineage import manifest_from_completion_evidence, validate_receipt_bound_evidence
from .proxy_classifier import (
    build_proxy_evaluation_binding,
    classify_proxy_outcome,
    validate_proxy_evaluation_binding,
)
from .phase_execution import AuthoritativePhaseContext, PhaseAuthorization
from .phase_receipts import validate_phase_run_receipt
from .phase_command_plan import validate_phase_command_plan
from .failure_validation import (
    canonical_evidence_bytes,
    evidence_bytes_hash,
    validate_failure_evidence as validate_failure_evidence_bytes,
    validate_resume_evidence as validate_resume_evidence_bytes,
)
from .utils import ensure_dir, now_utc

STATE_SCHEMA_VERSION = RESEARCH_STATE_SCHEMA_VERSION
ROUTE_OUTCOME_SCHEMA_VERSION = "auto_research_route_outcome_v4"
EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._/\-]{0,255}$")
ZERO_HASH = "0" * 64
_PROCESS_STATE_CACHE_LOCK = threading.RLock()
_PROCESS_STATE_CACHE: dict[str, tuple[int, str, dict[str, Any]]] = {}
TARGET_OUTCOMES = 5
ACTIVE_ATTEMPT_STATES = {"READY", "IMPLEMENTING", "IMPLEMENTATION_REPAIR", "PROXY_RUNNING", "PROXY_COMPLETED", "FULL_RUNNING", "RESOURCE_PAUSED"}
TERMINAL_ATTEMPT_STATES = {"METHOD_COMPLETED", "INTEGRITY_BLOCKED", "ABANDONED"}
EVENT_TYPES = {
    "DirectionSelected",
    "VariantPlanned",
    "AttemptReserved",
    "AttemptTransitioned",
    "ProxyPhaseStarted",
    "ProxyEvidenceCommitted",
    "FullPhaseStarted",
    "PhaseCommandStarted",
    "PhaseCommandCompleted",
    "PhaseCommandUnknownOutcome",
    "AttemptImplementationRevised",
    "AttemptResumed",
    "AttemptAbandoned",
    "AttemptDispositioned",
    "AttemptFinalized",
    "AuditMarker",
}
TRANSITIONS = {
    "PLANNED": {"IMPLEMENTING", "READY"},
    "IMPLEMENTING": {"READY"},
    "READY": {"PROXY_RUNNING", "FULL_RUNNING"},
    "PROXY_RUNNING": {"PROXY_COMPLETED"},
    "PROXY_COMPLETED": {"FULL_RUNNING"},
}

FAILURE_ROUTES = {
    "implementation_failure": ("REPAIR_IMPLEMENTATION", "IMPLEMENTATION_REPAIR"),
    "activation_failure": ("REPAIR_IMPLEMENTATION", "IMPLEMENTATION_REPAIR"),
    "resource_pause": ("PAUSE_RESOURCE", "RESOURCE_PAUSED"),
    "oom_retry": ("PAUSE_RESOURCE", "RESOURCE_PAUSED"),
    "integrity_failure": ("BLOCK_INTEGRITY", "INTEGRITY_BLOCKED"),
    "safety_failure": ("BLOCK_INTEGRITY", "INTEGRITY_BLOCKED"),
}


class IntegrityError(RuntimeError):
    """Raised when authoritative history or a requested transition is inconsistent."""


class _NoDomainEvent(RuntimeError):
    pass


class BreakingSchemaError(IntegrityError):
    """Raised when a v1 workspace is opened by the v2-only event store."""


def _begin_immediate(connection: sqlite3.Connection) -> None:
    for index in range(200):
        try:
            connection.execute("BEGIN IMMEDIATE")
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or index == 199:
                raise
            time.sleep(0.01)


class ResearchEventLedger:
    def __init__(self, project_root: Path, *, after_commit_hook: Callable[[], None] | None = None):
        self.project_root = Path(project_root)
        self.meta_dir = ensure_dir(self.project_root / "meta")
        self.db_path = self.meta_dir / "research_events.sqlite3"
        self.snapshot_path = self.meta_dir / "research_state.json"
        self.attempts_dir = ensure_dir(self.meta_dir / "attempts")
        self.route_path = self.meta_dir / "route_outcome.json"
        self.aggregate_path = self.meta_dir / "direction_outcome_aggregate.json"
        self.trial_path = self.project_root / "experiment" / "results" / "trial_result.json"
        self.trial_spec_path = self.project_root / "plan" / "trial_spec.json"
        self.after_commit_hook = after_commit_hook
        self._transaction_state_cache: dict[str, Any] | None = None
        self._transaction_cache_sequence = -1
        self._transaction_cache_event_hash = ""
        old_dir = self.meta_dir / "research_events"
        if old_dir.exists() and any(old_dir.glob("*.json")) and not self.db_path.exists():
            raise BreakingSchemaError("legacy event workspace is unsupported; rerun from S1 with Event v3")
        self._initialize_db()
        self._reject_legacy_workspace()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=60, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize_db(self) -> None:
        lock_path = self.meta_dir / ".research_events.init.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS events (
                        sequence INTEGER PRIMARY KEY,
                        event_id TEXT NOT NULL UNIQUE,
                        event_type TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        previous_event_hash TEXT NOT NULL,
                        event_hash TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL,
                        schema_version TEXT NOT NULL
                    )"""
                )
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _reject_legacy_workspace(self) -> None:
        if not self.db_path.exists():
            return
        with self._connect() as connection:
            versions = {row[0] for row in connection.execute("SELECT DISTINCT schema_version FROM events")}
        if versions and versions != {EVENT_SCHEMA_VERSION}:
            found = ", ".join(sorted(versions))
            raise BreakingSchemaError(f"event store schema {found} is unsupported; rerun from S1 with {EVENT_SCHEMA_VERSION}")

    def events(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            events = self._validated_events(connection)
            _reduce_all(self.project_root.name, events, project_root=self.project_root)
            return events

    def state(self) -> dict[str, Any]:
        with self._connect() as connection:
            state = self._state_in_transaction(connection, allow_cache=True)
        self._write_projections(state)
        return state

    def rebuild(self) -> dict[str, Any]:
        with self._connect() as connection:
            state = self._state_in_transaction(connection, allow_cache=False)
        self._write_projections(state)
        return state

    def query_operation_result(self, event_id: str) -> dict[str, Any]:
        """Return the immutable historical result derived at one committed event."""
        with self._connect() as connection:
            events = self._validated_events(connection)
        event = next((item for item in events if item["event_id"] == event_id), None)
        if event is None:
            raise IntegrityError(f"unknown event_id {event_id}")
        historical = _reduce_all(self.project_root.name, events[: event["sequence"]], project_root=self.project_root)
        payload = event["payload"]
        attempt_id = _event_attempt_id(payload)
        route = historical.get("last_route_outcome")
        if not isinstance(route, dict) or route.get("source", {}).get("event_id") != event_id:
            route = None
        return {
            "event_id": event["event_id"],
            "sequence": event["sequence"],
            "attempt_id": attempt_id,
            "event": deepcopy(event),
            "attempt": deepcopy(historical["attempts"].get(attempt_id)) if attempt_id else None,
            "route_outcome": deepcopy(route),
            "trial_result": deepcopy(historical["trial_results"].get(attempt_id)) if attempt_id else None,
            "aggregate": deepcopy(historical.get("latest_direction_aggregate")) if event["event_type"] == "AttemptFinalized" else None,
            "state_sequence": historical["last_sequence"],
        }

    def append(self, event_type: str, payload: dict[str, Any], *, event_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if event_type != "AuditMarker":
            raise IntegrityError("public append only permits AuditMarker; use a constrained domain method")
        return self._transact(event_type, payload, event_id or str(uuid.uuid4()))

    def _transact(self, event_type: str, payload: dict[str, Any], event_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        _validate_event_request(event_type, payload, event_id)
        payload_json = canonical_json(payload)
        with self._connect() as connection:
            _begin_immediate(connection)
            try:
                existing = connection.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
                if existing is not None:
                    if existing["event_type"] != event_type or existing["payload_json"] != payload_json:
                        raise IntegrityError(f"event_id conflict for {event_id}")
                    events = self._validated_events(connection)
                    state = self._state_from_events(events, allow_cache=True)
                    connection.commit()
                    event = _row_event(existing)
                else:
                    events = self._validated_events(connection)
                    state = self._state_from_events(events, allow_cache=True)
                    sequence = len(events) + 1
                    previous_hash = events[-1]["event_hash"] if events else ZERO_HASH
                    created_at = now_utc()
                    event = {
                        "schema_version": EVENT_SCHEMA_VERSION,
                        "event_id": event_id,
                        "sequence": sequence,
                        "event_type": event_type,
                        "previous_event_hash": previous_hash,
                        "created_at": created_at,
                        "payload": deepcopy(payload),
                    }
                    event["event_hash"] = _event_hash(event)
                    state["project_root"] = str(self.project_root)
                    next_state = reduce_event(state, event)
                    next_state.pop("project_root", None)
                    _validate_state_invariants(next_state)
                    connection.execute(
                        "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (sequence, event_id, event_type, payload_json, previous_hash, event["event_hash"], created_at, EVENT_SCHEMA_VERSION),
                    )
                    connection.commit()
                    state = next_state
                    self._remember_transaction_state(state, event)
            except Exception:
                connection.rollback()
                raise
        if self.after_commit_hook is not None:
            self.after_commit_hook()
        self._write_projections(state)
        return event, state

    def _domain_transact(
        self,
        build: Callable[[dict[str, Any]], tuple[str, dict[str, Any], str]],
        *,
        explicit_event_id: str | None = None,
        request_fingerprint: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        with self._connect() as connection:
            _begin_immediate(connection)
            try:
                events = self._validated_events(connection)
                state = self._state_from_events(events, allow_cache=True)
                if explicit_event_id is not None and request_fingerprint is not None:
                    existing = connection.execute("SELECT * FROM events WHERE event_id = ?", (explicit_event_id,)).fetchone()
                    if existing is not None:
                        event = _row_event(existing)
                        historical = event["payload"].get("request_fingerprint")
                        if historical != request_fingerprint:
                            raise IntegrityError(f"event_id request fingerprint conflict for {explicit_event_id}")
                        event_state = _reduce_all(self.project_root.name, events[: event["sequence"]], project_root=self.project_root)
                        connection.commit()
                        self._write_projections(state)
                        return event, event_state, True
                build_state = deepcopy(state)
                build_state["project_root"] = str(self.project_root)
                event_type, payload, event_id = build(build_state)
                if request_fingerprint is not None:
                    payload = {**payload, "request_fingerprint": request_fingerprint}
                _validate_event_request(event_type, payload, event_id)
                payload_json = canonical_json(payload)
                existing = connection.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
                if existing is not None:
                    if existing["event_type"] != event_type or existing["payload_json"] != payload_json:
                        raise IntegrityError(f"event_id conflict for {event_id}")
                    event = _row_event(existing)
                    event_state = _reduce_all(self.project_root.name, events[: event["sequence"]], project_root=self.project_root)
                    connection.commit()
                    replayed = True
                else:
                    sequence = len(events) + 1
                    previous_hash = events[-1]["event_hash"] if events else ZERO_HASH
                    created_at = now_utc()
                    event = {
                        "schema_version": EVENT_SCHEMA_VERSION,
                        "event_id": event_id,
                        "sequence": sequence,
                        "event_type": event_type,
                        "previous_event_hash": previous_hash,
                        "created_at": created_at,
                        "payload": deepcopy(payload),
                    }
                    event["event_hash"] = _event_hash(event)
                    state["project_root"] = str(self.project_root)
                    event_state = reduce_event(state, event)
                    event_state.pop("project_root", None)
                    connection.execute(
                        "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (sequence, event_id, event_type, payload_json, previous_hash, event["event_hash"], created_at, EVENT_SCHEMA_VERSION),
                    )
                    connection.commit()
                    replayed = False
                    self._remember_transaction_state(event_state, event)
            except Exception:
                connection.rollback()
                raise
        if self.after_commit_hook is not None and not replayed:
            self.after_commit_hook()
        self._write_projections(event_state)
        return event, event_state, replayed

    def select_direction(self, direction: dict[str, Any], *, event_id: str | None = None) -> dict[str, Any]:
        validate_direction_identity(direction)
        request_fingerprint = canonical_hash({"operation": "select_direction", "direction_spec_hash": direction["direction_spec_hash"]})
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            existing = state["directions"].get(direction["direction_semantic_hash"])
            if existing and existing["status"] in {"FINISHED", "EXHAUSTED"}:
                raise IntegrityError(f"closed direction semantic hash cannot be reopened: {direction['direction_semantic_hash']}")
            return "DirectionSelected", {"direction": direction}, event_id or f"direction:{direction['direction_spec_hash']}"

        _, state, _ = self._domain_transact(build, explicit_event_id=event_id, request_fingerprint=request_fingerprint if event_id else None)
        return state

    def plan_variant(self, variant: dict[str, Any], *, feedback_from_attempt_ids: list[str] | None = None, event_id: str | None = None) -> dict[str, Any]:
        payload = {"variant": variant, "feedback_from_attempt_ids": feedback_from_attempt_ids or []}
        request_fingerprint = canonical_hash({"operation": "plan_variant", **payload})
        _, state, _ = self._domain_transact(lambda _: ("VariantPlanned", payload, event_id or f"variant:{variant['variant_spec_hash']}"), explicit_event_id=event_id, request_fingerprint=request_fingerprint if event_id else None)
        return state

    def reserve_attempt(
        self,
        *,
        profile: str,
        direction: dict[str, Any],
        variant: dict[str, Any],
        implementation_hash: str,
        attempt_kind: str,
        trial_spec: dict[str, Any],
        event_id: str | None = None,
    ) -> dict[str, Any]:
        validate_variant_identity(direction, variant)
        validate_trial_spec(trial_spec)
        frozen_trial_spec = deepcopy(trial_spec)
        spec_hash = trial_spec_hash(frozen_trial_spec)
        input_hash = _attempt_input_hash_from_spec(implementation_hash, frozen_trial_spec)
        identity = canonical_hash({
            "profile": profile,
            "attempt_kind": attempt_kind,
            "direction_spec_hash": direction["direction_spec_hash"],
            "variant_spec_hash": variant["variant_spec_hash"],
            "attempt_input_hash": input_hash,
        })
        attempt_id = _canonical_attempt_id(self.project_root.name, identity)
        request_fingerprint = canonical_hash({
            "operation": "reserve_attempt", "profile": profile, "attempt_kind": attempt_kind,
            "direction_spec_hash": direction["direction_spec_hash"], "variant_spec_hash": variant["variant_spec_hash"],
            "implementation_hash": implementation_hash, "trial_spec_hash": spec_hash,
        })
        reservation_event_id = event_id or f"attempt:{profile}:{attempt_kind}:{identity}"

        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            _validate_trial_contracts(self.project_root, frozen_trial_spec)
            timestamp = now_utc()
            attempt = _build_canonical_attempt(
                project_id=state["project_id"],
                profile=profile,
                direction=direction,
                variant=variant,
                implementation_hash=implementation_hash,
                attempt_kind=attempt_kind,
                trial_spec=frozen_trial_spec,
                timestamp=timestamp,
            )
            validate_contract(attempt, "attempt_record_v8.schema.json")
            existing = state["attempts"].get(attempt_id)
            if existing is not None:
                immutable_keys = [
                    "profile", "attempt_kind", "direction_id", "direction_semantic_hash", "direction_spec_hash",
                    "variant_id", "variant_semantic_hash", "variant_spec_hash", "implementation_hash",
                    "trial_spec_hash", "acceptance_contract_hash", "attempt_input_hash",
                ]
                if any(existing[key] != attempt[key] for key in immutable_keys) or existing["frozen_trial_spec"] != frozen_trial_spec:
                    raise IntegrityError("attempt identity collision")
                return "AttemptReserved", {"attempt": existing}, reservation_event_id
            return "AttemptReserved", {"attempt": attempt}, reservation_event_id

        _, state, replayed = self._domain_transact(
            build, explicit_event_id=reservation_event_id, request_fingerprint=request_fingerprint
        )
        if replayed and event_id is None:
            state = self.state()
        return deepcopy(state["attempts"][attempt_id])

    def transition_attempt(self, attempt_id: str, new_state: str, *, phase: str | None = None, phase_state: str | None = None, event_id: str | None = None) -> dict[str, Any]:
        if new_state not in {"IMPLEMENTING", "READY"}:
            raise IntegrityError(f"public transition cannot enter authoritative failure/terminal state {new_state}")
        request_fingerprint = canonical_hash({"operation": "transition", "attempt_id": attempt_id, "new_state": new_state, "phase": phase, "phase_state": phase_state})
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            attempt = _attempt(state, attempt_id)
            payload = {"attempt_id": attempt_id, "lifecycle_generation": attempt["lifecycle_generation"], "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"], "expected_state": attempt["state"], "new_state": new_state, "phase": phase, "phase_state": phase_state}
            replay = state["operation_events"].get(_transition_replay_key(attempt, new_state, phase, phase_state))
            if replay is not None:
                return "AttemptTransitioned", replay["payload"], replay["event_id"]
            return "AttemptTransitioned", payload, event_id or _operation_event_id("transition", attempt, payload)
        _, state, _ = self._domain_transact(build, explicit_event_id=event_id, request_fingerprint=request_fingerprint if event_id else None)
        return deepcopy(state["attempts"][attempt_id])

    def start_proxy_phase(
        self,
        attempt_id: str,
        *,
        phase_execution_id: str,
        producer_run_id: str,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        return self._start_phase(
            attempt_id,
            phase="proxy",
            phase_execution_id=phase_execution_id,
            producer_run_id=producer_run_id,
            event_id=event_id,
        )

    def start_full_phase(
        self,
        attempt_id: str,
        *,
        phase_execution_id: str,
        producer_run_id: str,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        return self._start_phase(
            attempt_id,
            phase="full",
            phase_execution_id=phase_execution_id,
            producer_run_id=producer_run_id,
            event_id=event_id,
        )

    def _start_phase(
        self,
        attempt_id: str,
        *,
        phase: str,
        phase_execution_id: str,
        producer_run_id: str,
        event_id: str | None,
    ) -> dict[str, Any]:
        request_fingerprint = canonical_hash({"operation": "start_phase", "attempt_id": attempt_id, "phase": phase, "phase_execution_id": phase_execution_id, "producer_run_id": producer_run_id})
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            attempt = _attempt(state, attempt_id)
            _validate_trial_spec_projection(self.project_root, attempt)
            expected_state = "READY" if phase == "proxy" or attempt["attempt_kind"] == "full" else "PROXY_COMPLETED"
            if attempt["state"] != expected_state:
                raise IntegrityError(f"{phase} phase requires {expected_state}")
            if phase == "full":
                proxy = attempt.get("committed_proxy_outcome")
                if attempt["attempt_kind"] == "proxy_full" and (not isinstance(proxy, dict) or proxy.get("decision") != "RUN_FULL"):
                    raise IntegrityError("full phase requires committed RUN_FULL ProxyOutcome")
            operation_id = event_id or _operation_event_id(f"{phase}-phase-started", attempt, {"phase": phase, "phase_execution_id": phase_execution_id, "producer_run_id": producer_run_id})
            phase_contract = next(item for item in attempt["frozen_trial_spec"]["phase_contracts"] if item["phase"] == phase)
            manifest = {
                "schema_version": PHASE_EXECUTION_MANIFEST_SCHEMA_VERSION,
                "attempt_id": attempt["attempt_id"],
                "direction_semantic_hash": attempt["direction_semantic_hash"],
                "direction_spec_hash": attempt["direction_spec_hash"],
                "variant_semantic_hash": attempt["variant_semantic_hash"],
                "variant_spec_hash": attempt["variant_spec_hash"],
                "trial_spec_hash": attempt["trial_spec_hash"],
                "lifecycle_generation": attempt["lifecycle_generation"],
                "implementation_hash": attempt["implementation_hash"],
                "attempt_input_hash": attempt["attempt_input_hash"],
                "phase": phase,
                "phase_execution_id": phase_execution_id,
                "phase_start_event_id": operation_id,
                "producer_run_id": producer_run_id,
                "command_plan_hash": phase_contract["command_plan_hash"],
                "command_plan_ref": deepcopy(phase_contract["command_plan_ref"]),
                "phase_contract_hash": canonical_hash(phase_contract),
                "expected_evidence_kinds": sorted(phase_contract["evidence_kinds"]),
                "provenance_mode": phase_contract["command_plan"]["adapter_identity"]["provenance_mode"],
                "proxy_evaluation_binding": None,
                "proxy_authorization": None,
            }
            if phase == "proxy":
                manifest["proxy_evaluation_binding"] = build_proxy_evaluation_binding(
                    attempt=attempt,
                    phase_execution_id=phase_execution_id,
                    phase_start_event_id=operation_id,
                    producer_run_id=producer_run_id,
                    command_plan_hash=manifest["command_plan_hash"],
                    phase_contract_hash=manifest["phase_contract_hash"],
                    sample_contract_ref=attempt["frozen_trial_spec"]["sample_manifest"],
                    evaluator_contract_ref=attempt["frozen_trial_spec"]["execution_contract"]["evaluator_provenance"],
                    provenance_mode=manifest["provenance_mode"],
                )
            elif attempt["attempt_kind"] == "proxy_full":
                manifest["proxy_authorization"] = {
                    "proxy_event_id": proxy["event_id"],
                    "proxy_event_hash": proxy["event_hash"],
                    "proxy_outcome_hash": proxy["outcome_hash"],
                }
            validate_contract(manifest, "phase_execution_manifest_v3.schema.json")
            payload = {"attempt_id": attempt_id, "lifecycle_generation": attempt["lifecycle_generation"], "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"], "expected_state": expected_state, "phase_execution_manifest": manifest}
            return ("ProxyPhaseStarted" if phase == "proxy" else "FullPhaseStarted"), payload, operation_id
        _, state, _ = self._domain_transact(build, explicit_event_id=event_id, request_fingerprint=request_fingerprint if event_id else None)
        return deepcopy(state["attempts"][attempt_id])

    def commit_proxy_evidence(self, completion_evidence: dict[str, Any], *, event_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        _validate_completion_request_shape(completion_evidence)
        request_fingerprint = canonical_hash({"operation": "commit_proxy_evidence", "completion_evidence": completion_evidence})
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            attempt = _attempt(state, completion_evidence["attempt_id"])
            if attempt["attempt_kind"] != "proxy_full" or attempt["state"] != "PROXY_RUNNING":
                raise IntegrityError("ProxyEvidence commit requires a running proxy_full Attempt")
            manifest, _observations, completion_fingerprint = _decode_phase_completion(
                project_root=self.project_root, attempt=attempt, completion_evidence=completion_evidence,
                expected_phase="proxy", phase_commands=state["phase_commands"],
            )
            bound = _receipt_bound_phase_evidence(self.project_root, state, attempt, manifest, "proxy")
            proxy_outcome = _canonical_proxy_outcome_from_bound(attempt, bound)
            validate_contract(proxy_outcome, "proxy_outcome_v3.schema.json")
            payload = {"proxy_outcome": proxy_outcome, "evidence_manifest": manifest, "request_fingerprint": completion_fingerprint}
            return "ProxyEvidenceCommitted", payload, event_id or _operation_event_id("proxy-evidence-committed", attempt, payload)
        _, state, _ = self._domain_transact(build, explicit_event_id=event_id, request_fingerprint=request_fingerprint if event_id else None)
        attempt = state["attempts"][completion_evidence["attempt_id"]]
        return deepcopy(attempt), deepcopy(state["last_route_outcome"])

    def authorize_phase(self, context: AuthoritativePhaseContext) -> PhaseAuthorization:
        if not isinstance(context, AuthoritativePhaseContext):
            raise IntegrityError("phase authorization requires AuthoritativePhaseContext")
        with self._connect() as connection:
            events = self._validated_events(connection)
            state = self._state_from_events(events, allow_cache=True)
        authorization = _phase_authorization(state, events, context.attempt_id, context.phase)
        if authorization.authorization_hash != context.authorization_hash:
            raise IntegrityError("phase context does not match current SQLite authorization")
        return authorization

    def start_phase_command(self, command: dict[str, Any]) -> dict[str, Any]:
        validate_contract(command, "phase_command_v3.schema.json")
        request_fingerprint = canonical_hash({"operation": "start_phase_command", "command": command})

        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            existing = state["phase_commands"].get(command["command_id"])
            if existing is not None:
                _validate_command_replay(existing, command)
                return "PhaseCommandStarted", {"command": command}, existing["started_event_id"]
            _validate_phase_command_authorization(state, command)
            payload = {"command": deepcopy(command)}
            return "PhaseCommandStarted", payload, f"phase-command-started:{command['idempotency_key']}"

        event, state, _ = self._domain_transact(build)
        return _phase_command_operation_result(state["phase_commands"][command["command_id"]], event)

    def complete_phase_command(self, command_id: str, receipt_ref: dict[str, Any]) -> dict[str, Any]:
        request_fingerprint = canonical_hash({"operation": "complete_phase_command", "command_id": command_id, "receipt_ref": receipt_ref})

        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            record = _phase_command(state, command_id)
            command = record["command"]
            payload = {
                "command_id": command_id,
                "command_hash": command["command_hash"],
                "command_plan_hash": command["command_plan_hash"],
                "receipt_ref": deepcopy(receipt_ref),
                "receipt_hash": receipt_ref["digest"],
            }
            if record["status"] == "completed":
                if canonical_json(record["receipt_ref"]) != canonical_json(receipt_ref):
                    raise IntegrityError("completed phase command receipt conflict")
                return "PhaseCommandCompleted", payload, record["completed_event_id"]
            if record["status"] != "started":
                raise IntegrityError("only a started phase command can complete")
            _validate_phase_command_authorization(state, record["command"])
            _validate_phase_run_receipt(self.project_root, record, receipt_ref)
            return "PhaseCommandCompleted", payload, f"phase-command-completed:{request_fingerprint}"

        event, state, _ = self._domain_transact(build)
        return _phase_command_operation_result(state["phase_commands"][command_id], event)

    def mark_phase_command_unknown(self, command_id: str, reason: str) -> dict[str, Any]:
        if not isinstance(reason, str) or not reason.strip():
            raise IntegrityError("unknown command outcome requires a reason")
        reason = reason.strip()
        request_fingerprint = canonical_hash({"operation": "mark_phase_command_unknown", "command_id": command_id, "reason": reason})

        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            record = _phase_command(state, command_id)
            if record["status"] == "unknown":
                if record["unknown_reason"] != reason:
                    raise IntegrityError("unknown phase command reason conflict")
                return "PhaseCommandUnknownOutcome", {"command_id": command_id, "reason": reason}, record["unknown_event_id"]
            if record["status"] != "started":
                raise IntegrityError("only a started phase command can become unknown")
            _validate_phase_command_authorization(state, record["command"])
            return "PhaseCommandUnknownOutcome", {"command_id": command_id, "reason": reason}, f"phase-command-unknown:{request_fingerprint}"

        event, state, _ = self._domain_transact(build)
        return _phase_command_operation_result(state["phase_commands"][command_id], event)

    def phase_command(self, command_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            state = self._state_in_transaction(connection, allow_cache=True)
        record = state["phase_commands"].get(command_id)
        if isinstance(record, dict) and record.get("status") == "completed":
            _validate_phase_run_receipt(self.project_root, record, record["receipt_ref"])
        return deepcopy(record) if record is not None else None

    def revise_implementation(self, attempt_id: str, *, implementation_hash: str, event_id: str | None = None) -> dict[str, Any]:
        request_fingerprint = canonical_hash({"operation": "revise_implementation", "attempt_id": attempt_id, "implementation_hash": implementation_hash})
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            attempt = _attempt(state, attempt_id)
            _validate_trial_spec_projection(self.project_root, attempt)
            input_hash = _attempt_input_hash_for_implementation(attempt, implementation_hash)
            replay = state["operation_events"].get(_revision_replay_key(attempt, implementation_hash, input_hash))
            if replay is not None:
                return "AttemptImplementationRevised", replay["payload"], replay["event_id"]
            payload = _revision_payload(attempt, implementation_hash, input_hash)
            return "AttemptImplementationRevised", payload, event_id or _operation_event_id("implementation-revision", attempt, payload)
        _, state, _ = self._domain_transact(build, explicit_event_id=event_id, request_fingerprint=request_fingerprint if event_id else None)
        return deepcopy(state["attempts"][attempt_id])

    def abandon_attempt(self, attempt_id: str, *, reason: str, event_id: str | None = None) -> dict[str, Any]:
        request_fingerprint = canonical_hash({"operation": "abandon", "attempt_id": attempt_id, "reason": reason})
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            attempt = _attempt(state, attempt_id)
            payload = {"attempt_id": attempt_id, "lifecycle_generation": attempt["lifecycle_generation"], "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"], "expected_state": attempt["state"], "reason": reason}
            return "AttemptAbandoned", payload, event_id or _operation_event_id("attempt-abandoned", attempt, payload)
        _, state, _ = self._domain_transact(build, explicit_event_id=event_id, request_fingerprint=request_fingerprint if event_id else None)
        return deepcopy(state["attempts"][attempt_id])

    def disposition_failure(self, failure_evidence: dict[str, Any], *, event_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        validate_contract(failure_evidence, "failure_evidence_v6.schema.json")
        if str(failure_evidence.get("command_id") or "").startswith("core-resource-probe-"):
            raise IntegrityError("caller-authored Core resource receipts are forbidden")
        request_fingerprint = canonical_hash({"operation": "disposition_failure", "evidence": failure_evidence})
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            attempt = _attempt(state, failure_evidence["attempt_id"])
            payload = {"failure_evidence": deepcopy(failure_evidence)}
            key = _disposition_replay_key_from_evidence(failure_evidence)
            replay = state["operation_events"].get(key)
            if replay is not None:
                return "AttemptDispositioned", replay["payload"], replay["event_id"]
            _validate_failure_evidence(self.project_root, state, attempt, failure_evidence)
            return "AttemptDispositioned", payload, event_id or _operation_event_id("attempt-disposition", attempt, payload)
        _, event_state, _ = self._domain_transact(build, explicit_event_id=event_id, request_fingerprint=request_fingerprint if event_id else None)
        attempt_id = failure_evidence["attempt_id"]
        return deepcopy(event_state["attempts"][attempt_id]), deepcopy(event_state["last_route_outcome"])

    def pause_if_resources_unavailable(
        self,
        attempt_id: str,
        *,
        measurement_provider: Callable[[str, str, str], float] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Measure frozen phase resources inside the transaction and pause atomically."""

        result: tuple[dict[str, Any], dict[str, Any]] | None = None

        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            nonlocal result
            attempt = _attempt(state, attempt_id)
            phase = "proxy" if attempt["state"] == "PROXY_RUNNING" else "full" if attempt["state"] == "FULL_RUNNING" else None
            if phase is None:
                raise IntegrityError("resource measurement requires an authoritative running phase")
            execution = attempt["phase_executions"][phase]
            plan = ContractStore(self.project_root).read_json(execution["command_plan_hash"], schema_file="phase_command_plan_v2.schema.json")
            gpu_policies = [item["resource_policy"] for item in plan["commands"] if item["resource_policy"]["resource_class"] == "gpu_memory"]
            if not gpu_policies:
                result = None
                raise _NoDomainEvent()
            required = max(float(item["minimum_capacity"]) for item in gpu_policies)
            unit = gpu_policies[0]["unit"]
            provider = measurement_provider or _measure_os_resource
            observed = float(provider("gpu_memory", "gpu:any", unit))
            if observed >= required:
                result = None
                raise _NoDomainEvent()
            producer_run_id = execution["producer_run_id"]
            observed_at = now_utc()
            receipt = {
                "schema_version": "auto_research_core_resource_probe_receipt_v2",
                "operation": "pause",
                "attempt_id": attempt_id,
                **{
                    key: attempt[key]
                    for key in (
                        "direction_semantic_hash",
                        "direction_spec_hash",
                        "variant_semantic_hash",
                        "variant_spec_hash",
                        "trial_spec_hash",
                        "protocol_hash",
                        "sample_manifest_hash",
                        "evaluator_hash",
                    )
                },
                "lifecycle_generation": attempt["lifecycle_generation"],
                "implementation_hash": attempt["implementation_hash"],
                "attempt_input_hash": attempt["attempt_input_hash"],
                "phase": phase,
                "phase_execution_id": execution["phase_execution_id"],
                "phase_start_event_id": execution["phase_start_event_id"],
                "authority_event_id": execution["phase_start_event_id"],
                "producer_run_id": producer_run_id,
                "resource_type": "gpu_memory",
                "resource_id": "gpu:any",
                "required_capacity": required,
                "observed_capacity": observed,
                "unit": unit,
                "probe_status": "insufficient",
                "observed_at": observed_at,
                "provider_id": "constrained-test-resource-provider-v1" if measurement_provider else "core-os-resource-provider-v1",
            }
            receipt_ref = ContractStore(self.project_root).put_json(receipt, schema_file="core_resource_probe_receipt_v2.schema.json")
            command_id = f"core-resource-probe-{attempt_id[:12]}-g{attempt['lifecycle_generation']}"
            command_hash = canonical_hash({"command_id": command_id, "receipt": receipt_ref["digest"]})
            command_plan_hash = canonical_hash({"operation": "phase-resource-probe", "trial_spec_hash": attempt["trial_spec_hash"], "phase": phase, "required": required, "unit": unit})
            probe = {
                "schema_version": "auto_research_resource_probe_evidence_v4",
                "evidence_kind": "resource_probe",
                "evidence_id": f"resource-probe-{attempt_id[:12]}-g{attempt['lifecycle_generation']}",
                **{key: attempt[key] for key in ["attempt_id", "direction_semantic_hash", "direction_spec_hash", "variant_semantic_hash", "variant_spec_hash", "trial_spec_hash", "protocol_hash", "sample_manifest_hash", "evaluator_hash", "lifecycle_generation", "implementation_hash", "attempt_input_hash"]},
                "producer_run_id": producer_run_id,
                "phase": phase,
                "phase_execution_id": execution["phase_execution_id"],
                "phase_start_event_id": execution["phase_start_event_id"],
                "resource_type": "gpu_memory", "resource_id": "gpu:any", "required_capacity": required,
                "observed_capacity": observed, "unit": unit, "probe_status": "insufficient", "observed_at": observed_at,
                "command_id": command_id, "command_hash": command_hash, "command_plan_hash": command_plan_hash,
                "receipt_ref": receipt_ref, "receipt_hash": receipt_ref["digest"],
            }
            validate_contract(probe, "resource_probe_v4.schema.json")
            probe_hash = _write_operation_evidence(self.project_root, attempt, producer_run_id, "resource_probe", probe)
            failure = {
                "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION,
                "evidence_kind": "failure_evidence",
                "evidence_id": f"failure-resource-{attempt_id[:12]}-g{attempt['lifecycle_generation']}",
                **{key: attempt[key] for key in ["attempt_id", "direction_semantic_hash", "direction_spec_hash", "variant_semantic_hash", "variant_spec_hash", "trial_spec_hash", "protocol_hash", "sample_manifest_hash", "evaluator_hash", "lifecycle_generation", "implementation_hash", "attempt_input_hash"]},
                "producer_run_id": producer_run_id,
                "cross_references": {"resource_probe_hash": probe_hash},
                "source_state": attempt["state"], "source_phase": phase, "phase": phase,
                "phase_execution_id": execution["phase_execution_id"], "phase_start_event_id": execution["phase_start_event_id"],
                "failure_class": "resource_pause", "command_status": "resource_paused", "exit_code": 75,
                "reason": "frozen phase resource requirement is unavailable", "observed_at": observed_at,
                "log_hash": probe_hash,
                "command_id": command_id, "command_hash": command_hash, "command_plan_hash": command_plan_hash,
                "receipt_ref": receipt_ref, "receipt_hash": receipt_ref["digest"],
            }
            validate_contract(failure, "failure_evidence_v6.schema.json")
            _write_operation_evidence(self.project_root, attempt, producer_run_id, "failure_evidence", failure)
            payload = {"failure_evidence": failure}
            return "AttemptDispositioned", payload, _operation_event_id("attempt-disposition", attempt, payload)

        try:
            _, state, _ = self._domain_transact(build)
        except _NoDomainEvent:
            return None
        attempt = deepcopy(state["attempts"][attempt_id])
        result = (attempt, deepcopy(state["last_route_outcome"]))
        return result

    def resume_attempt(self, resume_evidence: dict[str, Any], *, event_id: str | None = None) -> dict[str, Any]:
        validate_contract(resume_evidence, "resume_evidence_v5.schema.json")
        if str(resume_evidence.get("command_id") or "").startswith("core-resource-"):
            raise IntegrityError("core resource resume evidence can only be created by resume_resource_attempt")
        request_fingerprint = canonical_hash({"operation": "resume_attempt", "evidence": resume_evidence})
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            attempt = _attempt(state, resume_evidence["attempt_id"])
            replay = state["operation_events"].get(_resume_replay_key(resume_evidence))
            if replay is not None:
                return "AttemptResumed", replay["payload"], replay["event_id"]
            _validate_resume_evidence(self.project_root, state, attempt, resume_evidence)
            payload = {"resume_evidence": deepcopy(resume_evidence)}
            return "AttemptResumed", payload, event_id or _operation_event_id("attempt-resumed", attempt, payload)
        _, state, _ = self._domain_transact(build, explicit_event_id=event_id, request_fingerprint=request_fingerprint if event_id else None)
        return deepcopy(state["attempts"][resume_evidence["attempt_id"]])

    def resume_resource_attempt(
        self,
        attempt_id: str,
        *,
        measurement_provider: Callable[[str, str, str], float] | None = None,
    ) -> dict[str, Any]:
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            attempt = _attempt(state, attempt_id)
            if attempt["state"] != "RESOURCE_PAUSED":
                raise IntegrityError("resource resume requires RESOURCE_PAUSED")
            pause_operation = _latest_pause_operation(state, attempt)
            if pause_operation is None:
                raise IntegrityError("resource resume requires a committed resource pause event")
            pause_event_id, pause_evidence = pause_operation
            pause_probe_raw = _read_canonical_operation_evidence(
                self.project_root,
                attempt,
                pause_evidence["producer_run_id"],
                "resource_probe",
                pause_evidence["cross_references"]["resource_probe_hash"],
            )
            pause_probe = json.loads(pause_probe_raw.decode("utf-8"))
            resource_type = pause_probe["resource_type"]
            resource_id = pause_probe["resource_id"]
            required = float(pause_probe["required_capacity"])
            unit = pause_probe["unit"]
            provider = measurement_provider or _measure_os_resource
            observed = float(provider(resource_type, resource_id, unit))
            if observed < required:
                raise IntegrityError("resource remains insufficient for resume")
            producer_run_id = f"resume-{attempt_id[:12]}-g{attempt['lifecycle_generation']}"
            phase_execution_id = f"resume-{attempt_id[:12]}-g{attempt['lifecycle_generation']}"
            observed_at = now_utc()
            receipt = {
                "schema_version": "auto_research_core_resource_probe_receipt_v2",
                "operation": "resume",
                "attempt_id": attempt_id,
                **{
                    key: attempt[key]
                    for key in (
                        "direction_semantic_hash",
                        "direction_spec_hash",
                        "variant_semantic_hash",
                        "variant_spec_hash",
                        "trial_spec_hash",
                        "protocol_hash",
                        "sample_manifest_hash",
                        "evaluator_hash",
                    )
                },
                "lifecycle_generation": attempt["lifecycle_generation"],
                "implementation_hash": attempt["implementation_hash"],
                "attempt_input_hash": attempt["attempt_input_hash"],
                "phase": "resume",
                "phase_execution_id": phase_execution_id,
                "phase_start_event_id": pause_event_id,
                "authority_event_id": pause_event_id,
                "producer_run_id": producer_run_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "required_capacity": required,
                "observed_capacity": observed,
                "unit": unit,
                "probe_status": "available",
                "observed_at": observed_at,
                "provider_id": "constrained-test-resource-provider-v1" if measurement_provider else "core-os-resource-provider-v1",
            }
            receipt_ref = ContractStore(self.project_root).put_json(
                receipt, schema_file="core_resource_probe_receipt_v2.schema.json"
            )
            command_id = f"core-resource-resume-{attempt_id[:12]}-g{attempt['lifecycle_generation']}"
            command_hash = canonical_hash({"command_id": command_id, "receipt": receipt_ref["digest"]})
            command_plan_hash = canonical_hash({
                "operation": "resume-resource-probe",
                "trial_spec_hash": attempt["trial_spec_hash"],
                "paused_phase": attempt.get("paused_phase"),
            })
            identity = {
                key: attempt[key]
                for key in [
                    "attempt_id", "direction_semantic_hash", "direction_spec_hash",
                    "variant_semantic_hash", "variant_spec_hash", "trial_spec_hash",
                    "protocol_hash", "sample_manifest_hash", "evaluator_hash",
                    "lifecycle_generation", "implementation_hash", "attempt_input_hash",
                ]
            }
            probe = {
                "schema_version": "auto_research_resource_probe_evidence_v4",
                "evidence_kind": "resource_probe",
                "evidence_id": f"resource-resume-{attempt_id[:12]}-g{attempt['lifecycle_generation']}",
                **identity,
                "producer_run_id": producer_run_id,
                "phase": "resume",
                "phase_execution_id": phase_execution_id,
                "phase_start_event_id": pause_event_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "required_capacity": required,
                "observed_capacity": observed,
                "unit": unit,
                "probe_status": "available",
                "observed_at": observed_at,
                "command_id": command_id,
                "command_hash": command_hash,
                "command_plan_hash": command_plan_hash,
                "receipt_ref": receipt_ref,
                "receipt_hash": receipt_ref["digest"],
            }
            validate_contract(probe, "resource_probe_v4.schema.json")
            probe_hash = _write_operation_evidence(
                self.project_root, attempt, producer_run_id, "resource_probe", probe
            )
            resume_evidence = {
                "schema_version": RESUME_EVIDENCE_SCHEMA_VERSION,
                "evidence_kind": "resume_evidence",
                "evidence_id": f"resume-{attempt_id[:12]}-g{attempt['lifecycle_generation']}",
                **identity,
                "producer_run_id": producer_run_id,
                "cross_references": {"resource_probe_hash": probe_hash},
                "phase": "resume",
                "phase_execution_id": phase_execution_id,
                "phase_start_event_id": pause_event_id,
                "pause_event_id": pause_event_id,
                "pause_evidence_hash": evidence_bytes_hash(canonical_evidence_bytes(pause_evidence)),
                "pause_phase": pause_evidence["phase"],
                "pause_phase_execution_id": pause_evidence["phase_execution_id"],
                "pause_producer_run_id": pause_evidence["producer_run_id"],
                "resource_type": resource_type,
                "resource_id": resource_id,
                "required_capacity": required,
                "observed_capacity": observed,
                "unit": unit,
                "probe_status": "available",
                "observed_at": observed_at,
                "command_id": command_id,
                "command_hash": command_hash,
                "command_plan_hash": command_plan_hash,
                "receipt_ref": receipt_ref,
                "receipt_hash": receipt_ref["digest"],
            }
            validate_contract(resume_evidence, "resume_evidence_v5.schema.json")
            _write_operation_evidence(
                self.project_root, attempt, producer_run_id, "resume_evidence", resume_evidence
            )
            _validate_resume_evidence(
                self.project_root, state, attempt, resume_evidence, allow_core_resource=True
            )
            payload = {"resume_evidence": resume_evidence}
            return "AttemptResumed", payload, _operation_event_id("attempt-resumed", attempt, payload)

        _, state, _ = self._domain_transact(build)
        return deepcopy(state["attempts"][attempt_id])

    def complete_attempt(self, completion_evidence: dict[str, Any], *, event_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        _validate_completion_request_shape(completion_evidence)
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            attempt = _attempt(state, completion_evidence["attempt_id"])
            _validate_trial_spec_projection(self.project_root, attempt)
            legal_state = (
                attempt["state"] == "METHOD_COMPLETED"
                or (attempt["attempt_kind"] in {"proxy", "bootstrap_proxy"} and attempt["state"] == "PROXY_RUNNING")
                or (attempt["attempt_kind"] in {"full", "proxy_full"} and attempt["state"] == "FULL_RUNNING")
            )
            if not legal_state:
                raise IntegrityError(f"attempt {attempt['attempt_id']} cannot finalize from {attempt['state']}")
            trial_result, completion_fingerprint = _classify_completion_evidence(
                project_root=self.project_root,
                attempt=attempt,
                completion_evidence=completion_evidence,
                phase_commands=state["phase_commands"],
            )
            replay = state["operation_events"].get(_finalization_attempt_key(attempt["attempt_id"]))
            if replay is not None:
                replay_payload = replay["payload"]
                if replay_payload.get("request_fingerprint") != completion_fingerprint:
                    raise IntegrityError("attempt completion fingerprint conflict")
                if replay_payload.get("trial_result") != trial_result:
                    raise IntegrityError("attempt completion replay result mismatch")
                if attempt["state"] != "METHOD_COMPLETED" or not attempt["method_evaluable"]:
                    raise IntegrityError("committed finalization does not match authoritative Attempt state")
                return "AttemptFinalized", replay["payload"], replay["event_id"]
            _validate_trial_against_state(state, trial_result)
            payload = {"trial_result": trial_result, "lifecycle_generation": attempt["lifecycle_generation"], "expected_state": attempt["state"], "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"], "request_fingerprint": completion_fingerprint}
            return "AttemptFinalized", payload, event_id or _operation_event_id("attempt-finalized", attempt, payload)
        _, event_state, _ = self._domain_transact(
            build,
            explicit_event_id=event_id,
            request_fingerprint=None,
        )
        return deepcopy(event_state["attempts"][completion_evidence["attempt_id"]]), deepcopy(event_state["last_route_outcome"])

    def validate_trial_precommit(self, completion_evidence: dict[str, Any]) -> dict[str, Any]:
        _validate_completion_request_shape(completion_evidence)
        with self._connect() as connection:
            state = self._state_in_transaction(connection)
        attempt = _attempt(state, completion_evidence["attempt_id"])
        _validate_trial_spec_projection(self.project_root, attempt)
        trial_result, _ = _classify_completion_evidence(
            project_root=self.project_root,
            attempt=attempt,
            completion_evidence=completion_evidence,
            phase_commands=state["phase_commands"],
        )
        _validate_trial_against_state(state, trial_result)
        return trial_result

    def _validated_events(self, connection: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = connection.execute("SELECT * FROM events ORDER BY sequence").fetchall()
        events: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        previous_hash = ZERO_HASH
        for expected_sequence, row in enumerate(rows, start=1):
            event = _row_event(row)
            _validate_event(event)
            if event["sequence"] != expected_sequence:
                raise IntegrityError(f"event sequence gap or duplicate at {event['sequence']}; expected {expected_sequence}")
            if event["event_id"] in seen_ids:
                raise IntegrityError(f"duplicate event_id {event['event_id']}")
            if event["previous_event_hash"] != previous_hash:
                raise IntegrityError(f"event hash chain mismatch at sequence {expected_sequence}")
            if event["event_hash"] != _event_hash(event):
                raise IntegrityError(f"event_hash mismatch at sequence {expected_sequence}")
            seen_ids.add(event["event_id"])
            previous_hash = event["event_hash"]
            events.append(event)
        return events

    def _state_in_transaction(self, connection: sqlite3.Connection, *, allow_cache: bool = True) -> dict[str, Any]:
        return self._state_from_events(self._validated_events(connection), allow_cache=allow_cache)

    def _state_from_events(self, events: list[dict[str, Any]], *, allow_cache: bool) -> dict[str, Any]:
        sequence = events[-1]["sequence"] if events else 0
        event_hash = events[-1]["event_hash"] if events else ZERO_HASH
        if (
            allow_cache
            and self._transaction_state_cache is not None
            and self._transaction_cache_sequence == sequence
            and self._transaction_cache_event_hash == event_hash
        ):
            self._validate_cached_authoritative_references(events, self._transaction_state_cache)
            cached_result = deepcopy(self._transaction_state_cache)
            cached_result.pop("project_root", None)
            return cached_result
        process_key = str(self.project_root.resolve())
        if allow_cache:
            with _PROCESS_STATE_CACHE_LOCK:
                process_cached = _PROCESS_STATE_CACHE.get(process_key)
            if process_cached is not None and process_cached[0] == sequence and process_cached[1] == event_hash:
                cached_state = process_cached[2]
                self._validate_cached_authoritative_references(events, cached_state)
                self._transaction_state_cache = deepcopy(cached_state)
                self._transaction_cache_sequence = sequence
                self._transaction_cache_event_hash = event_hash
                cached_result = deepcopy(cached_state)
                cached_result.pop("project_root", None)
                return cached_result
        state = _reduce_all(self.project_root.name, events, project_root=self.project_root)
        if allow_cache:
            self._transaction_state_cache = deepcopy(state)
            self._transaction_cache_sequence = sequence
            self._transaction_cache_event_hash = event_hash
            with _PROCESS_STATE_CACHE_LOCK:
                _PROCESS_STATE_CACHE[process_key] = (sequence, event_hash, deepcopy(state))
        return state

    def _validate_cached_authoritative_references(
        self, events: list[dict[str, Any]], state: dict[str, Any]
    ) -> None:
        for record in state.get("phase_commands", {}).values():
            if record.get("status") == "completed":
                _validate_phase_run_receipt(self.project_root, record, record["receipt_ref"])
        for event in events:
            payload = event.get("payload") or {}
            manifests: list[tuple[dict[str, Any], str]] = []
            if event.get("event_type") == "ProxyEvidenceCommitted":
                manifests.append((payload.get("evidence_manifest"), "proxy"))
            elif event.get("event_type") == "AttemptFinalized":
                trial = payload.get("trial_result") or {}
                manifests.append((trial.get("evidence_manifest"), str(trial.get("completeness") or "full")))
            for manifest, phase in manifests:
                attempt_id = (manifest or {}).get("attempt_id")
                attempt = (state.get("attempts") or {}).get(attempt_id)
                if not isinstance(attempt, dict):
                    raise IntegrityError("cached receipt-bound evidence Attempt is missing")
                audit_attempt = deepcopy(attempt)
                for key in (
                    "lifecycle_generation",
                    "implementation_hash",
                    "attempt_input_hash",
                    "trial_spec_hash",
                ):
                    if (manifest or {}).get(key) is not None:
                        audit_attempt[key] = (manifest or {})[key]
                try:
                    validate_receipt_bound_evidence(
                        project_root=self.project_root,
                        attempt=audit_attempt,
                        trial_spec=audit_attempt["frozen_trial_spec"],
                        manifest=manifest,
                        phase_commands=state.get("phase_commands") or {},
                        phase="proxy" if phase == "proxy" else "full",
                    )
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    raise IntegrityError(f"immutable receipt-bound evidence audit failed: {exc}") from exc

    def _remember_transaction_state(self, state: dict[str, Any], event: dict[str, Any]) -> None:
        cached = deepcopy(state)
        cached["project_root"] = str(self.project_root)
        self._transaction_state_cache = cached
        self._transaction_cache_sequence = event["sequence"]
        self._transaction_cache_event_hash = event["event_hash"]
        with _PROCESS_STATE_CACHE_LOCK:
            _PROCESS_STATE_CACHE[str(self.project_root.resolve())] = (
                event["sequence"],
                event["event_hash"],
                deepcopy(cached),
            )

    def _write_projections(self, state: dict[str, Any]) -> None:
        del state
        lock_path = self.meta_dir / ".research_projection.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            with self._connect() as connection:
                latest = self._state_in_transaction(connection, allow_cache=True)
            if self.snapshot_path.exists():
                try:
                    projected = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    self._quarantine_projection(self.snapshot_path)
                    projected = {}
                if int(projected.get("last_sequence", -1)) > latest["last_sequence"]:
                    self._quarantine_projection(self.snapshot_path)
            self._atomic_json(self.snapshot_path, latest)
            ensure_dir(self.attempts_dir)
            for stale in self.attempts_dir.glob("*.json"):
                stale.unlink()
            for attempt in latest["attempts"].values():
                self._atomic_json(self.attempts_dir / f"{attempt['attempt_id']}.json", attempt)
                self._atomic_json(_attempt_trial_spec_projection_path(self.project_root, attempt), attempt["frozen_trial_spec"])
            self._replace_projection(self.route_path, latest.get("last_route_outcome"))
            self._replace_projection(self.trial_path, latest.get("latest_trial_result"))
            self._replace_projection(self.aggregate_path, latest.get("latest_direction_aggregate"))
            active_attempts = [
                attempt for attempt in latest["attempts"].values()
                if attempt["state"] in ACTIVE_ATTEMPT_STATES
            ]
            if len(active_attempts) == 1:
                frozen = active_attempts[0]["frozen_trial_spec"]
            elif latest["attempts"]:
                latest_attempt = max(
                    latest["attempts"].values(),
                    key=lambda attempt: (attempt["updated_at"], attempt["attempt_id"]),
                )
                frozen = latest_attempt["frozen_trial_spec"]
            else:
                frozen = None
            if frozen is not None:
                self._replace_projection(self.trial_spec_path, frozen)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _quarantine_projection(self, path: Path) -> None:
        if path.exists():
            path.replace(path.with_name(f"{path.name}.quarantine.{uuid.uuid4().hex}"))

    def _replace_projection(self, path: Path, payload: dict[str, Any] | None) -> None:
        if payload is not None:
            self._atomic_json(path, payload)
        elif path.exists():
            path.unlink()

    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> None:
        ensure_dir(path.parent)
        file_descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def initial_state(project_id: str) -> dict[str, Any]:
    return {"schema_version": STATE_SCHEMA_VERSION, "project_id": project_id, "last_sequence": 0, "last_event_hash": ZERO_HASH, "directions": {}, "excluded_direction_semantic_hashes": [], "variants": {}, "attempts": {}, "trial_results": {}, "proxy_outcomes": {}, "phase_authorizations": {}, "phase_commands": {}, "method_tried_history": [], "implementation_history": [], "operation_events": {}, "current_direction_semantic_hash": None, "current_variant_spec_hash": None, "last_route_outcome": None, "latest_trial_result": None, "latest_direction_aggregate": None, "updated_at": None}


def _attempt_input_hash_from_spec(implementation_hash: str, trial_spec: dict[str, Any]) -> str:
    runtime_config = trial_spec["execution_contract"]["runtime_config"]
    runtime_config_hash = canonical_hash(runtime_config)
    if runtime_config_hash != trial_spec["execution_contract"]["runtime_config_hash"]:
        raise IntegrityError("TrialSpec runtime_config_hash mismatch")
    return attempt_input_hash(
        implementation_hash_value=implementation_hash,
        protocol=trial_spec["protocol"],
        sample_manifest=trial_spec["sample_manifest"],
        seeds=list(trial_spec["statistical_testing"]["seeds"]),
        runtime_config=runtime_config,
        evaluator_hash=trial_spec["execution_contract"]["evaluator_hash"],
        trial_spec=trial_spec,
    )


def _validate_trial_contracts(project_root: Path, trial_spec: dict[str, Any]) -> None:
    store = ContractStore(project_root)
    try:
        sample_manifest = store.read_contract(
            trial_spec["sample_manifest_ref"],
            contract_kind="sample_manifest",
            schema_file="sample_manifest_v4.schema.json",
        )
        evaluator_manifest = store.read_contract(
            trial_spec["execution_contract"]["evaluator_manifest_ref"],
            contract_kind="evaluator_manifest",
            schema_file="evaluator_manifest_v2.schema.json",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityError(f"TrialSpec ContractRef rejected: {exc}") from exc
    if canonical_contract_bytes(sample_manifest) != canonical_contract_bytes(trial_spec["sample_manifest"]):
        raise IntegrityError("sample manifest ContractRef content mismatch")
    if canonical_contract_bytes(evaluator_manifest) != canonical_contract_bytes(
        trial_spec["execution_contract"]["evaluator_provenance"]
    ):
        raise IntegrityError("evaluator manifest ContractRef content mismatch")
    if sample_manifest["provenance_mode"] != evaluator_manifest["provenance_mode"]:
        raise IntegrityError("sample and evaluator provenance modes must match")


def _canonical_attempt_id(project_id: str, identity: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{project_id}:{identity}"))


def _build_canonical_attempt(
    *,
    project_id: str,
    profile: str,
    direction: dict[str, Any],
    variant: dict[str, Any],
    implementation_hash: str,
    attempt_kind: str,
    trial_spec: dict[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    validate_variant_identity(direction, variant)
    validate_trial_spec(trial_spec)
    consumes, reserved = _profile_budget_mapping(profile, attempt_kind)
    frozen_trial_spec = deepcopy(trial_spec)
    input_hash = _attempt_input_hash_from_spec(implementation_hash, frozen_trial_spec)
    identity = canonical_hash(
        {
            "profile": profile,
            "attempt_kind": attempt_kind,
            "direction_spec_hash": direction["direction_spec_hash"],
            "variant_spec_hash": variant["variant_spec_hash"],
            "attempt_input_hash": input_hash,
        }
    )
    return {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_id": _canonical_attempt_id(project_id, identity),
        "profile": profile,
        "direction_id": direction["direction_id"],
        "direction_semantic_hash": direction["direction_semantic_hash"],
        "direction_spec_hash": direction["direction_spec_hash"],
        "variant_id": variant["variant_id"],
        "variant_semantic_hash": variant["variant_semantic_hash"],
        "variant_spec_hash": variant["variant_spec_hash"],
        "frozen_trial_spec": frozen_trial_spec,
        "trial_spec_hash": trial_spec_hash(frozen_trial_spec),
        "acceptance_contract_hash": acceptance_contract_hash(frozen_trial_spec),
        "implementation_hash": implementation_hash,
        "implementation_revisions": [
            {
                "previous_implementation_hash": None,
                "implementation_hash": implementation_hash,
                "attempt_input_hash": input_hash,
                "created_at": timestamp,
            }
        ],
        "attempt_input_hash": input_hash,
        "protocol_hash": canonical_hash(frozen_trial_spec["protocol"]),
        "sample_manifest_hash": canonical_hash(frozen_trial_spec["sample_manifest"]),
        "runtime_config_hash": frozen_trial_spec["execution_contract"]["runtime_config_hash"],
        "evaluator_hash": frozen_trial_spec["execution_contract"]["evaluator_hash"],
        "seeds": list(frozen_trial_spec["statistical_testing"]["seeds"]),
        "required_datasets": sorted(item["dataset_id"] for item in frozen_trial_spec["datasets"]),
        "required_phases": list(frozen_trial_spec["protocol"]["required_phases"]),
        "terminal_method_phases": list(frozen_trial_spec["protocol"]["terminal_phases"]),
        "required_roles": sorted(frozen_trial_spec["required_roles"]),
        "require_complete_seed_coverage": frozen_trial_spec["statistical_testing"]["require_complete_seed_coverage"],
        "attempt_kind": attempt_kind,
        "lifecycle_generation": 0,
        "state": "READY",
        "consumes_direction_budget": consumes,
        "reserved_slot": reserved,
        "phases": {"proxy": "PENDING", "full": "PENDING"},
        "phase_executions": {"proxy": None, "full": None},
        "committed_proxy_outcome": None,
        "paused_phase": None,
        "method_evaluable": False,
        "terminal_outcome": None,
        "failure_class": None,
        "artifact_hashes": {},
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def reduce_event(state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    if event["sequence"] != state["last_sequence"] + 1:
        raise IntegrityError(f"reducer expected sequence {state['last_sequence'] + 1}, got {event['sequence']}")
    if event["previous_event_hash"] != state["last_event_hash"]:
        raise IntegrityError("reducer previous_event_hash mismatch")
    next_state = deepcopy(state)
    payload = event["payload"]
    event_type = event["event_type"]
    if event_type == "DirectionSelected":
        direction = payload["direction"]
        validate_direction_identity(direction)
        semantic = direction["direction_semantic_hash"]
        existing = next_state["directions"].get(semantic)
        if existing and existing["status"] in {"FINISHED", "EXHAUSTED"}:
            raise IntegrityError(f"closed direction semantic hash cannot be reopened: {semantic}")
        if existing and existing["spec"]["direction_spec_hash"] != direction["direction_spec_hash"]:
            raise IntegrityError("active direction semantic identity cannot change spec")
        active = _active_standard_attempts(next_state)
        if active and any(item["direction_semantic_hash"] != semantic for item in active):
            raise IntegrityError("project execution_width=1 forbids selecting another direction")
        next_state["directions"].setdefault(semantic, {"spec": direction, "status": "ACTIVE", "budget": {"target": TARGET_OUTCOMES, "reserved": 0, "consumed": 0}, "aggregate": None})
        next_state["current_direction_semantic_hash"] = semantic
    elif event_type == "VariantPlanned":
        variant = payload["variant"]
        semantic = variant["direction_semantic_hash"]
        direction_state = _active_direction(next_state, semantic)
        if any(item.get("method_evaluable") and item.get("consumes_direction_budget", True) and item.get("variant_semantic_hash") == variant.get("variant_semantic_hash") for item in next_state["method_tried_history"]):
            raise IntegrityError("variant duplicates a method-evaluable scientific method in the current direction")
        validate_variant_identity(direction_state["spec"], variant, tried_variants=next_state["method_tried_history"])
        budget = direction_state["budget"]
        if budget["consumed"] + budget["reserved"] >= TARGET_OUTCOMES:
            raise IntegrityError("direction budget has no capacity for another variant")
        active = _active_standard_attempts(next_state)
        if active:
            raise IntegrityError("project execution_width=1 forbids planning while a standard attempt is active")
        next_state["variants"][variant["variant_spec_hash"]] = variant
        next_state["current_variant_spec_hash"] = variant["variant_spec_hash"]
    elif event_type == "AttemptReserved":
        attempt = payload["attempt"]
        validate_contract(attempt, "attempt_record_v8.schema.json")
        project_root = next_state.get("project_root")
        if not project_root:
            raise IntegrityError("AttemptReserved rebuild requires authoritative project_root")
        _validate_trial_contracts(Path(project_root), attempt["frozen_trial_spec"])
        expected_consumes, expected_reserved = _profile_budget_mapping(attempt["profile"], attempt["attempt_kind"])
        if attempt["consumes_direction_budget"] != expected_consumes or attempt["reserved_slot"] != expected_reserved:
            raise IntegrityError("attempt budget flags do not match profile/attempt_kind")
        if attempt["lifecycle_generation"] != 0:
            raise IntegrityError("new attempt lifecycle_generation must be zero")
        direction = _active_direction(next_state, attempt["direction_semantic_hash"])
        direction_spec = direction["spec"]
        for key in ["direction_id", "direction_semantic_hash", "direction_spec_hash"]:
            if attempt[key] != direction_spec[key]:
                raise IntegrityError(f"AttemptReserved {key} mismatch")
        variant = next_state["variants"].get(attempt["variant_spec_hash"])
        if not variant:
            raise IntegrityError("attempt references unknown VariantSpec")
        for key in ["variant_id", "variant_semantic_hash", "variant_spec_hash"]:
            if attempt[key] != variant[key]:
                raise IntegrityError(f"AttemptReserved {key} mismatch")
        canonical_attempt = _build_canonical_attempt(
            project_id=next_state["project_id"],
            profile=attempt["profile"],
            direction=direction_spec,
            variant=variant,
            implementation_hash=attempt["implementation_hash"],
            attempt_kind=attempt["attempt_kind"],
            trial_spec=attempt["frozen_trial_spec"],
            timestamp=attempt["created_at"],
        )
        if canonical_json(attempt) != canonical_json(canonical_attempt):
            raise IntegrityError("AttemptReserved record differs from canonical initial Attempt")
        _validate_frozen_trial_spec(attempt)
        if attempt["attempt_id"] in next_state["attempts"]:
            raise IntegrityError("duplicate attempt_id")
        if _active_standard_attempts(next_state):
            raise IntegrityError("project execution_width=1 reservation conflict")
        if attempt["reserved_slot"]:
            budget = direction["budget"]
            if budget["consumed"] + budget["reserved"] >= TARGET_OUTCOMES:
                raise IntegrityError("direction budget exhausted before reservation")
            budget["reserved"] += 1
        next_state["attempts"][attempt["attempt_id"]] = attempt
    elif event_type == "AttemptTransitioned":
        attempt = _attempt(next_state, payload["attempt_id"])
        _validate_operation_identity(attempt, payload)
        if payload["expected_state"] != attempt["state"]:
            raise IntegrityError("attempt transition source state mismatch")
        _apply_transition(attempt, payload["new_state"], payload.get("phase"), payload.get("phase_state"), event["created_at"])
        next_state["operation_events"][_transition_replay_key_from_payload(payload)] = {"event_id": event["event_id"], "payload": deepcopy(payload)}
    elif event_type in {"ProxyPhaseStarted", "FullPhaseStarted"}:
        attempt = _attempt(next_state, payload["attempt_id"])
        _validate_operation_identity(attempt, payload)
        if payload["expected_state"] != attempt["state"]:
            raise IntegrityError("phase start source state mismatch")
        manifest = payload["phase_execution_manifest"]
        validate_contract(manifest, "phase_execution_manifest_v3.schema.json")
        phase = manifest["phase"]
        if event_type == "ProxyPhaseStarted" and phase != "proxy":
            raise IntegrityError("ProxyPhaseStarted requires proxy manifest")
        if event_type == "FullPhaseStarted" and phase != "full":
            raise IntegrityError("FullPhaseStarted requires full manifest")
        for key in ("attempt_id", "lifecycle_generation", "implementation_hash", "attempt_input_hash"):
            if manifest[key] != attempt[key]:
                raise IntegrityError(f"phase manifest {key} mismatch")
        if manifest["phase_start_event_id"] != event["event_id"]:
            raise IntegrityError("phase manifest source event mismatch")
        if phase == "proxy":
            if attempt["state"] != "READY" or attempt["phases"]["proxy"] != "PENDING":
                raise IntegrityError("proxy start requires READY/PENDING")
            binding = manifest.get("proxy_evaluation_binding")
            if not isinstance(binding, dict):
                raise IntegrityError("proxy phase requires Ledger-generated evaluation binding")
            try:
                validate_proxy_evaluation_binding(
                    binding,
                    policy=attempt["frozen_trial_spec"]["proxy_decision_policy"],
                    attempt=attempt,
                )
            except ValueError as exc:
                raise IntegrityError(f"proxy evaluation binding rejected: {exc}") from exc
            if binding["phase_start_event_id"] != event["event_id"]:
                raise IntegrityError("proxy evaluation binding source event mismatch")
            attempt["state"] = "PROXY_RUNNING"
        else:
            if attempt["attempt_kind"] == "proxy_full":
                proxy = attempt.get("committed_proxy_outcome")
                if attempt["state"] != "PROXY_COMPLETED" or not isinstance(proxy, dict) or proxy["decision"] != "RUN_FULL":
                    raise IntegrityError("full start requires committed RUN_FULL proxy outcome")
            elif attempt["state"] != "READY":
                raise IntegrityError("full start requires READY")
            attempt["state"] = "FULL_RUNNING"
        attempt["phases"][phase] = "RUNNING"
        attempt["phase_executions"][phase] = deepcopy(manifest)
        next_state["phase_authorizations"][_phase_authorization_key(attempt["attempt_id"], phase)] = {
            "attempt_id": attempt["attempt_id"],
            "phase": phase,
            "phase_start_event_id": event["event_id"],
            "phase_start_event_hash": event["event_hash"],
            "phase_start_sequence": event["sequence"],
        }
        attempt["updated_at"] = event["created_at"]
    elif event_type == "PhaseCommandStarted":
        command = payload["command"]
        validate_contract(command, "phase_command_v3.schema.json")
        if command["command_id"] in next_state["phase_commands"]:
            raise IntegrityError("duplicate PhaseCommandStarted command_id")
        _validate_phase_command_authorization(next_state, command)
        next_state["phase_commands"][command["command_id"]] = {
            "command": deepcopy(command),
            "status": "started",
            "started_event_id": event["event_id"],
            "started_event_hash": event["event_hash"],
            "started_sequence": event["sequence"],
            "created_at": event["created_at"],
            "receipt_ref": None,
            "completed_event_id": None,
            "completed_event_hash": None,
            "completed_sequence": None,
            "unknown_event_id": None,
            "unknown_event_hash": None,
            "unknown_sequence": None,
            "unknown_reason": None,
            "updated_at": event["created_at"],
        }
    elif event_type == "PhaseCommandCompleted":
        record = _phase_command(next_state, payload["command_id"])
        if record["status"] != "started":
            raise IntegrityError("PhaseCommandCompleted requires started command")
        if payload["command_hash"] != record["command"]["command_hash"]:
            raise IntegrityError("PhaseCommandCompleted command_hash mismatch")
        if payload["command_plan_hash"] != record["command"]["command_plan_hash"]:
            raise IntegrityError("PhaseCommandCompleted command_plan_hash mismatch")
        if payload["receipt_hash"] != payload["receipt_ref"]["digest"]:
            raise IntegrityError("PhaseCommandCompleted receipt_hash mismatch")
        _validate_phase_command_authorization(next_state, record["command"])
        project_root = next_state.get("project_root")
        if not project_root:
            raise IntegrityError("PhaseCommandCompleted rebuild requires authoritative project_root")
        _validate_phase_run_receipt(Path(project_root), record, payload["receipt_ref"])
        record.update(
            status="completed",
            receipt_ref=deepcopy(payload["receipt_ref"]),
            completed_event_id=event["event_id"],
            completed_event_hash=event["event_hash"],
            completed_sequence=event["sequence"],
            updated_at=event["created_at"],
        )
    elif event_type == "PhaseCommandUnknownOutcome":
        record = _phase_command(next_state, payload["command_id"])
        if record["status"] != "started":
            raise IntegrityError("PhaseCommandUnknownOutcome requires started command")
        _validate_phase_command_authorization(next_state, record["command"])
        record.update(
            status="unknown",
            unknown_event_id=event["event_id"],
            unknown_event_hash=event["event_hash"],
            unknown_sequence=event["sequence"],
            unknown_reason=payload["reason"],
            updated_at=event["created_at"],
        )
        attempt = _attempt(next_state, record["command"]["attempt_id"])
        if attempt["state"] in TERMINAL_ATTEMPT_STATES:
            raise IntegrityError("unknown command outcome cannot modify a terminal Attempt")
        phase = record["command"]["phase"]
        attempt["state"] = "INTEGRITY_BLOCKED"
        attempt["failure_class"] = "integrity_failure"
        attempt["terminal_outcome"] = "integrity_blocked"
        attempt["method_evaluable"] = False
        attempt["phases"][phase] = "FAILED"
        attempt["updated_at"] = event["created_at"]
        if attempt["reserved_slot"]:
            budget = next_state["directions"][attempt["direction_semantic_hash"]]["budget"]
            budget["reserved"] -= 1
            attempt["reserved_slot"] = False
        route = build_route_outcome(
            next_state,
            "BLOCK_INTEGRITY",
            ["phase_command_unknown_outcome"],
            attempt,
            source_event_id=event["event_id"],
            source_sequence=event["sequence"],
        )
        next_state["last_route_outcome"] = route
        next_state["operation_events"][_command_unknown_replay_key(record["command"])] = {
            "event_id": event["event_id"],
            "payload": deepcopy(payload),
            "route_outcome": deepcopy(route),
        }
    elif event_type == "ProxyEvidenceCommitted":
        proxy_outcome = payload["proxy_outcome"]
        validate_contract(proxy_outcome, "proxy_outcome_v3.schema.json")
        attempt = _attempt(next_state, proxy_outcome["attempt_id"])
        if attempt["state"] != "PROXY_RUNNING" or attempt["phases"]["proxy"] != "RUNNING":
            raise IntegrityError("proxy commit requires running proxy phase")
        phase_execution = attempt["phase_executions"]["proxy"]
        for key in ("lifecycle_generation", "implementation_hash", "attempt_input_hash", "phase_execution_id", "phase_start_event_id"):
            if proxy_outcome[key] != phase_execution[key]:
                raise IntegrityError(f"ProxyOutcome {key} mismatch")
        if proxy_outcome["evidence_manifest_hash"] != canonical_hash(payload["evidence_manifest"]):
            raise IntegrityError("ProxyOutcome evidence manifest hash mismatch")
        project_root = next_state.get("project_root")
        if not project_root:
            raise IntegrityError("proxy evidence reduction requires project root")
        bound = _receipt_bound_phase_evidence(
            Path(project_root), next_state, attempt, payload["evidence_manifest"], "proxy"
        )
        expected_proxy = _canonical_proxy_outcome_from_bound(attempt, bound)
        if canonical_json(proxy_outcome) != canonical_json(expected_proxy):
            raise IntegrityError("ProxyOutcome differs from immutable evidence-derived outcome")
        attempt["phases"]["proxy"] = "COMPLETED"
        attempt["state"] = "PROXY_COMPLETED" if proxy_outcome["decision"] == "RUN_FULL" else "ABANDONED"
        attempt["committed_proxy_outcome"] = {"event_id": event["event_id"], "event_hash": event["event_hash"], "outcome_hash": canonical_hash(proxy_outcome), "decision": proxy_outcome["decision"]}
        attempt["updated_at"] = event["created_at"]
        next_state["proxy_outcomes"][attempt["attempt_id"]] = deepcopy(proxy_outcome)
        if proxy_outcome["decision"] != "RUN_FULL" and attempt["reserved_slot"]:
            next_state["directions"][attempt["direction_semantic_hash"]]["budget"]["reserved"] -= 1
            attempt["reserved_slot"] = False
        action = "RUN_FULL" if proxy_outcome["decision"] == "RUN_FULL" else proxy_outcome["decision"]
        next_state["last_route_outcome"] = build_route_outcome(next_state, action, proxy_outcome["reason_codes"], attempt, source_event_id=event["event_id"], source_sequence=event["sequence"])
    elif event_type == "AttemptImplementationRevised":
        attempt = _attempt(next_state, payload["attempt_id"])
        _validate_operation_identity(attempt, payload, require_input=False)
        if payload["expected_state"] != attempt["state"]:
            raise IntegrityError("implementation revision source state mismatch")
        if attempt["state"] != "IMPLEMENTATION_REPAIR":
            raise IntegrityError("implementation revision requires IMPLEMENTATION_REPAIR")
        if payload["implementation_hash"] == attempt["implementation_hash"]:
            raise IntegrityError("implementation repair requires a new implementation_hash")
        previous = attempt["implementation_hash"]
        attempt["implementation_hash"] = payload["implementation_hash"]
        attempt["attempt_input_hash"] = payload["attempt_input_hash"]
        attempt["implementation_revisions"].append({"previous_implementation_hash": previous, "implementation_hash": payload["implementation_hash"], "attempt_input_hash": payload["attempt_input_hash"], "created_at": event["created_at"]})
        attempt["lifecycle_generation"] += 1
        attempt["state"] = "READY"
        attempt["phases"] = {"proxy": "PENDING", "full": "PENDING"}
        attempt["phase_executions"] = {"proxy": None, "full": None}
        attempt["committed_proxy_outcome"] = None
        next_state["proxy_outcomes"].pop(attempt["attempt_id"], None)
        attempt["failure_class"] = None
        attempt["artifact_hashes"] = {}
        attempt["updated_at"] = event["created_at"]
        next_state["implementation_history"].append({"attempt_id": attempt["attempt_id"], "previous_implementation_hash": previous, "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"]})
        next_state["operation_events"][_revision_replay_key_from_payload(payload)] = {"event_id": event["event_id"], "payload": deepcopy(payload)}
    elif event_type == "AttemptResumed":
        evidence = payload["resume_evidence"]
        validate_contract(evidence, "resume_evidence_v5.schema.json")
        attempt = _attempt(next_state, evidence["attempt_id"])
        project_root = next_state.get("project_root")
        if not project_root:
            raise IntegrityError("resume reduction requires project root")
        _validate_resume_evidence(Path(project_root), next_state, attempt, evidence)
        if attempt["state"] != "RESOURCE_PAUSED":
            raise IntegrityError("resume requires RESOURCE_PAUSED")
        paused_phase = attempt.get("paused_phase")
        if paused_phase not in {"proxy", "full"} or attempt["phases"][paused_phase] != "RUNNING":
            raise IntegrityError("resource resume requires a running paused phase")
        attempt["lifecycle_generation"] += 1
        attempt["phases"][paused_phase] = "PENDING"
        attempt["phase_executions"][paused_phase] = None
        if paused_phase == "proxy":
            attempt["state"] = "READY"
            attempt["phases"]["full"] = "PENDING"
            attempt["phase_executions"]["full"] = None
            attempt["committed_proxy_outcome"] = None
            next_state["proxy_outcomes"].pop(attempt["attempt_id"], None)
        elif attempt["attempt_kind"] == "proxy_full":
            proxy = attempt.get("committed_proxy_outcome")
            if attempt["phases"]["proxy"] != "COMPLETED" or not isinstance(proxy, dict) or proxy.get("decision") != "RUN_FULL":
                raise IntegrityError("full resource resume requires committed RUN_FULL proxy authorization")
            attempt["state"] = "PROXY_COMPLETED"
        else:
            attempt["state"] = "READY"
        attempt["paused_phase"] = None
        attempt["failure_class"] = None
        attempt["artifact_hashes"] = {}
        attempt["updated_at"] = event["created_at"]
        next_state["operation_events"][_resume_replay_key(evidence)] = {"event_id": event["event_id"], "payload": deepcopy(payload)}
    elif event_type == "AttemptAbandoned":
        attempt = _attempt(next_state, payload["attempt_id"])
        _validate_operation_identity(attempt, payload)
        if payload["lifecycle_generation"] != attempt["lifecycle_generation"] or payload["expected_state"] != attempt["state"]:
            raise IntegrityError("attempt abandonment operation identity mismatch")
        if attempt["state"] in TERMINAL_ATTEMPT_STATES:
            raise IntegrityError("terminal attempt cannot be abandoned")
        if attempt["reserved_slot"]:
            next_state["directions"][attempt["direction_semantic_hash"]]["budget"]["reserved"] -= 1
        attempt["reserved_slot"] = False
        attempt["state"] = "ABANDONED"
        attempt["failure_class"] = payload["reason"]
        attempt["updated_at"] = event["created_at"]
    elif event_type == "AttemptDispositioned":
        evidence = payload["failure_evidence"]
        validate_contract(evidence, "failure_evidence_v6.schema.json")
        attempt = _attempt(next_state, evidence["attempt_id"])
        project_root = next_state.get("project_root")
        if not project_root:
            raise IntegrityError("failure reduction requires project root")
        _validate_failure_evidence(
            Path(project_root),
            next_state,
            attempt,
            evidence,
            allow_core_resource=str(evidence.get("command_id") or "").startswith("core-resource-probe-"),
        )
        action, target_state = _failure_route(evidence["failure_class"])
        _apply_failure_disposition(next_state, attempt, target_state, evidence, event["created_at"])
        route = build_route_outcome(next_state, action, [evidence["failure_class"]], attempt, source_event_id=event["event_id"], source_sequence=event["sequence"])
        next_state["last_route_outcome"] = route
        next_state["operation_events"][_disposition_replay_key_from_evidence(evidence)] = {"event_id": event["event_id"], "payload": deepcopy(payload)}
    elif event_type == "AttemptFinalized":
        trial = payload["trial_result"]
        attempt = _attempt(next_state, trial["attempt_id"])
        _validate_operation_identity(attempt, payload)
        if payload["expected_state"] != attempt["state"]:
            raise IntegrityError("attempt finalization source state mismatch")
        attempt = _validate_trial_against_state(next_state, trial)
        project_root = next_state.get("project_root")
        if not project_root:
            raise IntegrityError("TrialResult reduction requires project root")
        bound = _receipt_bound_phase_evidence(
            Path(project_root), next_state, attempt, trial["evidence_manifest"], trial["completeness"]
        )
        expected_trial = _canonical_trial_from_bound(attempt, bound)
        if canonical_json(trial) != canonical_json(expected_trial):
            raise IntegrityError("TrialResult differs from immutable evidence-derived result")
        next_state = _apply_trial_to_state(next_state, trial)
        expected_aggregate = _build_direction_aggregate(next_state, attempt["direction_semantic_hash"]) if next_state["directions"][attempt["direction_semantic_hash"]]["budget"]["consumed"] == TARGET_OUTCOMES else None
        route = _route_after_verified_trial(next_state, attempt, expected_aggregate, source_event_id=event["event_id"], source_sequence=event["sequence"])
        if expected_aggregate:
            next_state["directions"][attempt["direction_semantic_hash"]]["aggregate"] = expected_aggregate
            next_state["directions"][attempt["direction_semantic_hash"]]["status"] = "FINISHED" if route["next_action"] == "FINISH_DIRECTION" else "EXHAUSTED"
            next_state["latest_direction_aggregate"] = expected_aggregate
            next_state["excluded_direction_semantic_hashes"] = sorted(set(next_state["excluded_direction_semantic_hashes"] + [attempt["direction_semantic_hash"]]))
            next_state["current_direction_semantic_hash"] = None
            next_state["current_variant_spec_hash"] = None
        next_state["last_route_outcome"] = route
        next_state["latest_trial_result"] = trial
        next_state["operation_events"][_finalization_replay_key(trial)] = {"event_id": event["event_id"], "payload": deepcopy(payload)}
        next_state["operation_events"][_finalization_attempt_key(attempt["attempt_id"])] = {"event_id": event["event_id"], "payload": deepcopy(payload)}
    elif event_type == "AuditMarker":
        pass
    else:
        raise IntegrityError(f"unknown event_type {event_type}")
    next_state["last_sequence"] = event["sequence"]
    next_state["last_event_hash"] = event["event_hash"]
    next_state["updated_at"] = event["created_at"]
    _validate_state_invariants(next_state)
    return next_state


def build_route_outcome(state: dict[str, Any], next_action: str, reason_codes: list[str], attempt: dict[str, Any], *, source_event_id: str | None, source_sequence: int | None) -> dict[str, Any]:
    budget = deepcopy(state["directions"][attempt["direction_semantic_hash"]]["budget"])
    route = {"schema_version": ROUTE_OUTCOME_SCHEMA_VERSION, "source": {"event_id": source_event_id, "sequence": source_sequence, "attempt_id": attempt["attempt_id"]}, "identity": {key: attempt[key] for key in ["direction_id", "direction_semantic_hash", "direction_spec_hash", "variant_id", "variant_semantic_hash", "variant_spec_hash", "attempt_id"]}, "next_action": next_action, "reason_codes": list(reason_codes), "budget_snapshot": budget, "artifact_hashes": deepcopy(attempt.get("artifact_hashes") or {}), "idempotency_key": canonical_hash({"source_event_id": source_event_id, "source_sequence": source_sequence, "attempt_id": attempt["attempt_id"], "lifecycle_generation": attempt["lifecycle_generation"], "next_action": next_action, "reason_codes": list(reason_codes), "budget": budget, "artifact_hashes": attempt.get("artifact_hashes") or {}, "variant_spec_hash": attempt["variant_spec_hash"]})}
    validate_contract(route, "route_outcome_v4.schema.json")
    return route


def _route_after_verified_trial(state: dict[str, Any], attempt: dict[str, Any], aggregate: dict[str, Any] | None, *, source_event_id: str, source_sequence: int) -> dict[str, Any]:
    if attempt["profile"] == "bootstrap":
        if attempt["attempt_kind"] != "bootstrap_proxy":
            raise IntegrityError("bootstrap FINISH_RUN requires bootstrap_proxy attempt kind")
        return build_route_outcome(state, "FINISH_RUN", ["bootstrap_proxy_verified"], state["attempts"][attempt["attempt_id"]], source_event_id=source_event_id, source_sequence=source_sequence)
    budget = state["directions"][attempt["direction_semantic_hash"]]["budget"]
    if budget["consumed"] < TARGET_OUTCOMES:
        return build_route_outcome(state, "PROPOSE_NEXT_VARIANT", ["verified_outcome_recorded", "direction_budget_remaining"], state["attempts"][attempt["attempt_id"]], source_event_id=source_event_id, source_sequence=source_sequence)
    if aggregate is None:
        raise IntegrityError("fifth outcome requires DirectionOutcomeAggregate")
    action = "FINISH_DIRECTION" if any(item["outcome"] == "accepted" for item in aggregate["outcomes"]) else "START_NEW_DIRECTION"
    return build_route_outcome(state, action, ["five_verified_outcomes", aggregate["selection"]["status"]], state["attempts"][attempt["attempt_id"]], source_event_id=source_event_id, source_sequence=source_sequence)


def _apply_trial_to_state(state: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    next_state = deepcopy(state)
    attempt = _validate_trial_against_state(next_state, trial)
    if attempt["reserved_slot"]:
        budget = next_state["directions"][attempt["direction_semantic_hash"]]["budget"]
        if budget["consumed"] + budget["reserved"] > TARGET_OUTCOMES or budget["consumed"] >= TARGET_OUTCOMES:
            raise IntegrityError("budget exhausted during AttemptFinalized")
        budget["reserved"] -= 1
        budget["consumed"] += 1
        attempt["reserved_slot"] = False
    attempt["state"] = "METHOD_COMPLETED"
    attempt["method_evaluable"] = True
    attempt["terminal_outcome"] = trial["outcome_classification"]
    attempt["failure_class"] = None
    attempt["artifact_hashes"] = deepcopy(trial["raw_artifacts"])
    attempt["phases"][trial["completeness"]] = "COMPLETED"
    next_state["trial_results"][attempt["attempt_id"]] = trial
    if attempt["consumes_direction_budget"]:
        next_state["method_tried_history"].append({"attempt_id": attempt["attempt_id"], "profile": attempt["profile"], "consumes_direction_budget": True, "direction_semantic_hash": attempt["direction_semantic_hash"], "direction_spec_hash": attempt["direction_spec_hash"], "variant_id": attempt["variant_id"], "variant_semantic_hash": attempt["variant_semantic_hash"], "variant_spec_hash": attempt["variant_spec_hash"], "method_evaluable": True, "outcome_classification": trial["outcome_classification"], "trial_result_hash": canonical_hash(trial), "primary_metric_summary": trial["primary_metric_summary"]})
    return next_state


def _validate_trial_against_state(state: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    attempt = _attempt(state, trial["attempt_id"])
    try:
        validate_trial_result(trial, attempt=attempt, trial_spec=attempt["frozen_trial_spec"])
    except ValueError as exc:
        raise IntegrityError(f"AttemptFinalized canonical TrialResult mismatch: {exc}") from exc
    if attempt["state"] in TERMINAL_ATTEMPT_STATES or attempt["method_evaluable"]:
        existing = state["trial_results"].get(attempt["attempt_id"])
        if existing == trial:
            return attempt
        raise IntegrityError("attempt already finalized")
    for key in ["direction_id", "direction_semantic_hash", "direction_spec_hash", "variant_id", "variant_semantic_hash", "variant_spec_hash", "trial_spec_hash", "acceptance_contract_hash", "attempt_input_hash", "protocol_hash", "lifecycle_generation", "implementation_hash"]:
        if trial.get(key) != attempt.get(key):
            raise IntegrityError(f"TrialResult {key} mismatch")
    if not attempt["reserved_slot"] and attempt["consumes_direction_budget"]:
        raise IntegrityError("standard attempt has no reserved slot")
    if attempt["state"] not in {"PROXY_RUNNING", "PROXY_COMPLETED", "FULL_RUNNING"}:
        raise IntegrityError(f"attempt state {attempt['state']} cannot finalize")
    if trial["completeness"] == "proxy" and attempt["attempt_kind"] not in {"proxy", "bootstrap_proxy"}:
        raise IntegrityError("proxy cannot be terminal for this attempt kind")
    if trial["completeness"] == "full" and attempt["attempt_kind"] not in {"full", "proxy_full"}:
        raise IntegrityError("full result cannot finalize this attempt kind")
    if attempt["attempt_kind"] == "proxy_full":
        proxy = attempt.get("committed_proxy_outcome")
        if not isinstance(proxy, dict) or proxy.get("decision") != "RUN_FULL":
            raise IntegrityError("proxy_full finalization requires committed RUN_FULL ProxyOutcome")
        if trial.get("proxy_outcome_event_id") != proxy["event_id"] or trial.get("proxy_outcome_hash") != proxy["outcome_hash"]:
            raise IntegrityError("TrialResult ProxyOutcome binding mismatch")
    expected_state = "FULL_RUNNING" if trial["completeness"] == "full" else {"PROXY_RUNNING", "PROXY_COMPLETED"}
    if isinstance(expected_state, str) and attempt["state"] != expected_state:
        raise IntegrityError(f"{trial['completeness']} TrialResult requires {expected_state} execution state")
    if isinstance(expected_state, set) and attempt["state"] not in expected_state:
        raise IntegrityError("proxy TrialResult requires proxy execution state")
    phase_state = attempt["phases"][trial["completeness"]]
    if phase_state not in {"RUNNING", "COMPLETED"}:
        raise IntegrityError(f"TrialResult cannot finalize phase in {phase_state} state")
    _validate_trial_observations(attempt, trial)
    if attempt["consumes_direction_budget"] and any(item.get("consumes_direction_budget") and item["variant_semantic_hash"] == attempt["variant_semantic_hash"] and item["attempt_id"] != attempt["attempt_id"] for item in state["method_tried_history"]):
        raise IntegrityError("variant semantic hash already has a method-evaluable outcome")
    return attempt


def _validate_trial_observations(attempt: dict[str, Any], trial: dict[str, Any]) -> None:
    validate_trial_evidence(trial, attempt=attempt)
    phase_contract = next(item for item in attempt["frozen_trial_spec"]["phase_contracts"] if item["phase"] == trial["completeness"])
    required = set(phase_contract["datasets"])
    observed = set(trial["observed_datasets"])
    if set(trial["required_datasets"]) != required or required != observed:
        raise IntegrityError("required and observed dataset coverage mismatch")
    registered_artifacts = set(trial["raw_artifacts"].values())
    identities: set[tuple[Any, ...]] = set()
    roles_by_dataset_seed: dict[tuple[str, int], set[str]] = {}
    observed_seeds: set[int] = set()
    observation_datasets: set[str] = set()
    for observation in trial["observations"]:
        if observation["sample_manifest_hash"] != attempt["sample_manifest_hash"]:
            raise IntegrityError("observation sample manifest hash mismatch")
        if observation["evaluator_hash"] != attempt["evaluator_hash"]:
            raise IntegrityError("observation evaluator hash mismatch")
        if observation["seed"] not in attempt["seeds"]:
            raise IntegrityError("observation seed is not pre-registered")
        if observation["command_status"] != "completed":
            raise IntegrityError("method-evaluable observation command status must be completed")
        if observation["phase"] != trial["completeness"]:
            raise IntegrityError("observation phase disagrees with TrialResult completeness")
        phase_execution = attempt["phase_executions"][trial["completeness"]]
        for key in ("lifecycle_generation", "implementation_hash", "attempt_input_hash", "phase_execution_id", "phase_start_event_id", "producer_run_id"):
            if observation[key] != phase_execution[key]:
                raise IntegrityError(f"observation {key} mismatch")
        if observation["raw_artifact_hash"] not in registered_artifacts:
            raise IntegrityError("observation references an unregistered raw artifact hash")
        identity = (observation["phase"], observation["phase_execution_id"], observation["role"], observation["dataset_id"], observation["metric_id"], observation["seed"])
        if identity in identities:
            raise IntegrityError("duplicate observation identity")
        identities.add(identity)
        observed_seeds.add(observation["seed"])
        observation_datasets.add(observation["dataset_id"])
        roles_by_dataset_seed.setdefault((observation["dataset_id"], observation["seed"]), set()).add(observation["role"])
    if observation_datasets != required:
        raise IntegrityError("observation dataset coverage mismatch")
    if attempt["require_complete_seed_coverage"] and observed_seeds != set(phase_contract["seeds"]):
        raise IntegrityError("observation seed coverage mismatch")
    for dataset_id in required:
        for seed in phase_contract["seeds"]:
            if attempt["require_complete_seed_coverage"] and not set(phase_contract["roles"]).issubset(roles_by_dataset_seed.get((dataset_id, seed), set())):
                raise IntegrityError("pre-registered role coverage is incomplete")
    if trial["evidence_manifest_hash"] != canonical_hash(trial["evidence_manifest"]):
        raise IntegrityError("TrialResult evidence_manifest_hash mismatch")
    if trial["completeness"] not in attempt["terminal_method_phases"]:
        raise IntegrityError("TrialResult phase is not a pre-registered terminal method phase")


def _build_direction_aggregate(state: dict[str, Any], semantic: str) -> dict[str, Any]:
    outcomes = [item for item in state["method_tried_history"] if item["direction_semantic_hash"] == semantic and item.get("consumes_direction_budget")]
    if len(outcomes) != TARGET_OUTCOMES or len({item["attempt_id"] for item in outcomes}) != TARGET_OUTCOMES or len({item["variant_semantic_hash"] for item in outcomes}) != TARGET_OUTCOMES:
        raise IntegrityError("DirectionOutcomeAggregate requires exactly five unique attempts and semantic variants")
    rows = [{key: item[key] for key in ["attempt_id", "variant_id", "variant_semantic_hash", "variant_spec_hash", "trial_result_hash", "outcome_classification", "primary_metric_summary"]} for item in outcomes]
    for row in rows:
        row["outcome"] = row.pop("outcome_classification")
    accepted = [row for row in rows if row["outcome"] == "accepted"]
    comparable = [row for row in accepted if isinstance(row["primary_metric_summary"].get("delta"), (int, float))]
    if comparable and len({row["primary_metric_summary"].get("metric_id") for row in comparable}) == 1 and len({row["primary_metric_summary"].get("objective") for row in comparable}) == 1:
        objective = comparable[0]["primary_metric_summary"]["objective"]
        ordered = sorted(comparable, key=lambda row: (-row["primary_metric_summary"]["delta"], row["variant_semantic_hash"], row["variant_spec_hash"], row["attempt_id"]))
        selection = {"status": "selected", "best_attempt_id": ordered[0]["attempt_id"], "reason": f"best accepted pre-registered {objective} delta with deterministic semantic/spec tie-break"}
    else:
        selection = {"status": "inconclusive", "best_attempt_id": None, "reason": "insufficient comparable pre-registered primary metric evidence"}
    aggregate = {"schema_version": DIRECTION_AGGREGATE_SCHEMA_VERSION, "direction_semantic_hash": semantic, "direction_spec_hash": state["directions"][semantic]["spec"]["direction_spec_hash"], "status": "finished" if any(row["outcome"] == "accepted" for row in rows) else "exhausted", "outcomes": rows, "selection": selection}
    validate_contract(aggregate, "direction_outcome_aggregate_v1.schema.json")
    return aggregate


def _validate_event_request(event_type: str, payload: dict[str, Any], event_id: str) -> None:
    if event_type not in EVENT_TYPES:
        raise IntegrityError(f"unknown event_type {event_type}")
    if not EVENT_ID_PATTERN.fullmatch(event_id):
        raise IntegrityError(f"invalid event_id format: {event_id}")
    if not isinstance(payload, dict):
        raise IntegrityError("event payload must be an object")
    required = {"DirectionSelected": {"direction"}, "VariantPlanned": {"variant", "feedback_from_attempt_ids"}, "AttemptReserved": {"attempt"}, "AttemptTransitioned": {"attempt_id", "lifecycle_generation", "implementation_hash", "attempt_input_hash", "expected_state", "new_state", "phase", "phase_state"}, "ProxyPhaseStarted": {"attempt_id", "lifecycle_generation", "implementation_hash", "attempt_input_hash", "expected_state", "phase_execution_manifest"}, "FullPhaseStarted": {"attempt_id", "lifecycle_generation", "implementation_hash", "attempt_input_hash", "expected_state", "phase_execution_manifest"}, "ProxyEvidenceCommitted": {"proxy_outcome", "evidence_manifest"}, "PhaseCommandStarted": {"command"}, "PhaseCommandCompleted": {"command_id", "command_hash", "command_plan_hash", "receipt_ref", "receipt_hash"}, "PhaseCommandUnknownOutcome": {"command_id", "reason"}, "AttemptImplementationRevised": {"attempt_id", "lifecycle_generation", "previous_implementation_hash", "previous_attempt_input_hash", "expected_state", "implementation_hash", "attempt_input_hash"}, "AttemptResumed": {"resume_evidence"}, "AttemptAbandoned": {"attempt_id", "lifecycle_generation", "implementation_hash", "attempt_input_hash", "expected_state", "reason"}, "AttemptDispositioned": {"failure_evidence"}, "AttemptFinalized": {"trial_result", "lifecycle_generation", "expected_state", "implementation_hash", "attempt_input_hash"}, "AuditMarker": {"index"}}[event_type]
    actual = set(payload) - {"request_fingerprint"}
    if actual != required or ("request_fingerprint" in payload and not re.fullmatch(r"[a-f0-9]{64}", payload["request_fingerprint"])):
        raise IntegrityError(f"{event_type} payload fields must be {sorted(required)}")


def _validate_event(event: dict[str, Any]) -> None:
    if event.get("schema_version") != EVENT_SCHEMA_VERSION:
        raise BreakingSchemaError("invalid or unsupported Event v3 schema")
    try:
        validate_contract(event, "event_v8.schema.json")
    except ValueError as exc:
        raise IntegrityError(f"invalid Event v3 schema: {exc}") from exc
    _validate_event_request(event["event_type"], event["payload"], event["event_id"])


def _event_hash(event: dict[str, Any]) -> str:
    return canonical_hash({key: event[key] for key in ["schema_version", "event_id", "sequence", "event_type", "previous_event_hash", "created_at", "payload"]})


def _row_event(row: sqlite3.Row) -> dict[str, Any]:
    return {"schema_version": row["schema_version"], "event_id": row["event_id"], "sequence": row["sequence"], "event_type": row["event_type"], "previous_event_hash": row["previous_event_hash"], "event_hash": row["event_hash"], "created_at": row["created_at"], "payload": json.loads(row["payload_json"])}


def _event_attempt_id(payload: dict[str, Any]) -> str | None:
    if isinstance(payload.get("attempt"), dict):
        return payload["attempt"].get("attempt_id")
    if isinstance(payload.get("failure_evidence"), dict):
        return payload["failure_evidence"].get("attempt_id")
    if isinstance(payload.get("resume_evidence"), dict):
        return payload["resume_evidence"].get("attempt_id")
    if isinstance(payload.get("trial_result"), dict):
        return payload["trial_result"].get("attempt_id")
    return payload.get("attempt_id")


def _reduce_all(project_id: str, events: list[dict[str, Any]], *, project_root: Path | None = None) -> dict[str, Any]:
    state = initial_state(project_id)
    state["project_root"] = str(project_root) if project_root is not None else None
    for event in events:
        state = reduce_event(state, event)
    state.pop("project_root", None)
    return state


def _receipt_bound_phase_evidence(
    project_root: Path,
    state: dict[str, Any],
    attempt: dict[str, Any],
    manifest: dict[str, Any],
    phase: str,
):
    try:
        bound = validate_receipt_bound_evidence(
            project_root=project_root,
            attempt=attempt,
            trial_spec=attempt["frozen_trial_spec"],
            manifest=manifest,
            phase_commands=state["phase_commands"],
            phase=phase,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise IntegrityError(f"immutable receipt-bound evidence audit failed: {exc}") from exc
    if canonical_json(bound.manifest) != canonical_json(manifest):
        raise IntegrityError("event EvidenceManifest differs from canonical receipt-derived lineage")
    return bound


def _canonical_trial_from_bound(attempt: dict[str, Any], bound: Any) -> dict[str, Any]:
    try:
        return classify_trial_result(
            attempt=attempt,
            trial_spec=attempt["frozen_trial_spec"],
            evidence_manifest=bound.manifest,
            evidence_bytes=bound.evidence_bytes,
        )
    except (ValueError, OSError) as exc:
        raise IntegrityError(f"immutable TrialResult audit failed: {exc}") from exc


def _canonical_proxy_outcome_from_bound(attempt: dict[str, Any], bound: Any) -> dict[str, Any]:
    phase_execution = attempt.get("phase_executions", {}).get("proxy") or {}
    binding = phase_execution.get("proxy_evaluation_binding")
    if not isinstance(binding, dict):
        raise IntegrityError("proxy phase lacks authoritative evaluation binding")
    try:
        return classify_proxy_outcome(
            frozen_policy=attempt["frozen_trial_spec"]["proxy_decision_policy"],
            evaluation_binding=binding,
            decoded_evidence=bound.decoded_evidence,
            evidence_manifest_hash=canonical_hash(bound.manifest),
        )
    except ValueError as exc:
        raise IntegrityError(f"immutable ProxyOutcome audit failed: {exc}") from exc


def _phase_authorization_key(attempt_id: str, phase: str) -> str:
    return f"{attempt_id}:{phase}"


def _phase_authorization(
    state: dict[str, Any],
    events: list[dict[str, Any]],
    attempt_id: str,
    phase: str,
) -> PhaseAuthorization:
    del events
    attempt = _attempt(state, attempt_id)
    if phase not in {"proxy", "full"}:
        raise IntegrityError("phase authorization requires proxy or full")
    expected_state = "PROXY_RUNNING" if phase == "proxy" else "FULL_RUNNING"
    if attempt["state"] != expected_state or attempt["phases"][phase] != "RUNNING":
        raise IntegrityError(f"{phase} phase is not currently authorized")
    manifest = attempt["phase_executions"].get(phase)
    source = state["phase_authorizations"].get(_phase_authorization_key(attempt_id, phase))
    if not isinstance(manifest, dict) or not isinstance(source, dict):
        raise IntegrityError("authoritative phase execution is missing")
    if source["phase_start_event_id"] != manifest["phase_start_event_id"]:
        raise IntegrityError("phase authorization source event mismatch")
    runtime = attempt["frozen_trial_spec"]["execution_contract"]["runtime_config"]
    collector = str(runtime.get("collector") or "generic")
    adapter_identity = "adapter-" + re.sub(r"[^A-Za-z0-9_-]", "-", collector).strip("-")
    if len(adapter_identity) < 8:
        adapter_identity = "adapter-generic"
    adapter_identity = adapter_identity[:127]
    provenance_mode = manifest["provenance_mode"].replace("-", "_")
    proxy = attempt.get("committed_proxy_outcome") if attempt["attempt_kind"] == "proxy_full" and phase == "full" else None
    try:
        return PhaseAuthorization(
            attempt_id=attempt["attempt_id"],
            lifecycle_generation=attempt["lifecycle_generation"],
            phase=phase,
            phase_execution_id=manifest["phase_execution_id"],
            phase_start_event_id=source["phase_start_event_id"],
            phase_start_event_hash=source["phase_start_event_hash"],
            phase_start_sequence=source["phase_start_sequence"],
            producer_run_id=manifest["producer_run_id"],
            implementation_hash=attempt["implementation_hash"],
            attempt_input_hash=attempt["attempt_input_hash"],
            trial_spec_hash=attempt["trial_spec_hash"],
            command_plan_hash=manifest["command_plan_hash"],
            phase_contract_hash=manifest["phase_contract_hash"],
            expected_evidence_kinds=tuple(manifest["expected_evidence_kinds"]),
            adapter_identity=adapter_identity,
            provenance_mode=provenance_mode,
            state=attempt["state"],
            proxy_authorization_required=phase != "full" or attempt["attempt_kind"] == "proxy_full",
            proxy_commit_event_id=proxy.get("event_id") if isinstance(proxy, dict) else None,
            proxy_commit_event_hash=proxy.get("event_hash") if isinstance(proxy, dict) else None,
            proxy_outcome_hash=proxy.get("outcome_hash") if isinstance(proxy, dict) else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityError(f"phase authorization is malformed: {exc}") from exc


def _validate_phase_command_authorization(state: dict[str, Any], command: dict[str, Any]) -> None:
    authorization = _phase_authorization(state, [], command["attempt_id"], command["phase"])
    expected = {
        "attempt_id": authorization.attempt_id,
        "lifecycle_generation": authorization.lifecycle_generation,
        "phase": authorization.phase,
        "phase_execution_id": authorization.phase_execution_id,
        "phase_start_event_id": authorization.phase_start_event_id,
        "producer_run_id": authorization.producer_run_id,
        "implementation_hash": authorization.implementation_hash,
        "attempt_input_hash": authorization.attempt_input_hash,
        "authorization_hash": authorization.authorization_hash,
        "provenance_mode": authorization.provenance_mode.replace("_", "-"),
    }
    for key, value in expected.items():
        if command.get(key) != value:
            raise IntegrityError(f"PhaseCommand {key} differs from SQLite authorization")
    attempt = _attempt(state, command["attempt_id"])
    phase_contract = next(
        (
            item
            for item in attempt["frozen_trial_spec"]["phase_contracts"]
            if item["phase"] == command["phase"]
        ),
        None,
    )
    if not isinstance(phase_contract, dict):
        raise IntegrityError("PhaseCommand phase is not preregistered")
    if command["command_plan_hash"] != phase_contract["command_plan_hash"]:
        raise IntegrityError("PhaseCommand command_plan_hash differs from frozen phase contract")
    try:
        store = ContractStore(Path(state["project_root"]))
        frozen_plan = store.read_json(command["command_plan_ref"], schema_file="phase_command_plan_v2.schema.json")
        validate_phase_command_plan(frozen_plan, expected_evidence_kinds=phase_contract["evidence_kinds"])
        plan_blob = store.verify(command["command_plan_ref"])
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise IntegrityError(f"PhaseCommandPlan rejected: {exc}") from exc
    if plan_blob["digest"] != command["command_plan_hash"]:
        raise IntegrityError("PhaseCommandPlan reference hash mismatch")
    matches = [item for item in frozen_plan["commands"] if item["command_spec_id"] == command["command_spec_id"]]
    if len(matches) != 1 or canonical_json(matches[0]) != canonical_json(command["command_spec"]):
        raise IntegrityError("PhaseCommand does not exactly match its frozen command spec")
    command_spec = command["command_spec"]
    if contract_digest(canonical_contract_bytes(command_spec)) != command["command_hash"]:
        raise IntegrityError("PhaseCommand command_hash mismatch")
    execution_records = [
        record
        for record in state["phase_commands"].values()
        if record["command"]["attempt_id"] == command["attempt_id"]
        and record["command"]["lifecycle_generation"] == command["lifecycle_generation"]
        and record["command"]["phase_execution_id"] == command["phase_execution_id"]
        and record["command"]["command_id"] != command["command_id"]
    ]
    if any(record["command"]["command_spec_id"] == command["command_spec_id"] for record in execution_records):
        raise IntegrityError("duplicate PhaseCommand command_spec_id")
    completed_specs = {
        record["command"]["command_spec_id"]
        for record in execution_records
        if record["status"] == "completed"
    }
    if any(dependency not in completed_specs for dependency in command_spec["dependencies"]):
        raise IntegrityError("PhaseCommand dependencies are not completed")
    if any(record["status"] != "completed" for record in execution_records):
        raise IntegrityError("a prior PhaseCommand is not authoritatively completed")
    records_by_spec = {record["command"]["command_spec_id"]: record for record in execution_records}

    def completed_receipt(spec_id: str) -> dict[str, Any] | None:
        record = records_by_spec.get(spec_id)
        if not isinstance(record, dict) or record.get("status") != "completed":
            return None
        receipt_ref = record.get("receipt_ref")
        if not isinstance(receipt_ref, dict):
            return None
        return _validate_phase_run_receipt(Path(state["project_root"]), record, receipt_ref)

    def receipt_is_oom(receipt: dict[str, Any] | None) -> bool:
        if not isinstance(receipt, dict) or int(receipt.get("exit_code", 0)) == 0:
            return False
        try:
            stderr = ContractStore(Path(state["project_root"])).read_bytes(receipt["stderr_ref"]).decode("utf-8").lower()
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            raise IntegrityError(f"conditional command receipt log rejected: {exc}") from exc
        return any(token in stderr for token in ("out of memory", "cuda oom", "cublas_status_alloc_failed"))

    spec_id = command_spec["command_spec_id"]
    if spec_id == "full-train_recovery_reduced_concurrency" and not receipt_is_oom(completed_receipt("full-train")):
        raise IntegrityError("reduced-concurrency recovery requires an authoritative OOM train receipt")
    if spec_id == "full-train_recovery_memory_safe":
        predecessor = "full-train_recovery_reduced_concurrency" if any(
            item["command_spec_id"] == "full-train_recovery_reduced_concurrency" for item in frozen_plan["commands"]
        ) else "full-train"
        if not receipt_is_oom(completed_receipt(predecessor)):
            raise IntegrityError("memory-safe recovery requires an authoritative predecessor OOM receipt")
    if spec_id.startswith(("full-eval_", "full-ablation_eval_", "full-derive-evidence")):
        training_receipts = [
            completed_receipt(item["command_spec_id"])
            for item in frozen_plan["commands"]
            if item["command_spec_id"] == "full-train" or item["command_spec_id"].startswith("full-train_recovery_")
        ]
        if not any(isinstance(receipt, dict) and int(receipt["exit_code"]) == 0 for receipt in training_receipts):
            raise IntegrityError("full post-training command requires one successful authoritative train receipt")
    preceding = [item for item in frozen_plan["commands"] if item["ordinal"] < command_spec["ordinal"]]
    missing_unconditional = [
        item["command_spec_id"]
        for item in preceding
        if item["condition"]["kind"] == "always" and item["command_spec_id"] not in completed_specs
    ]
    if missing_unconditional:
        raise IntegrityError("PhaseCommand is out of frozen plan order")
    expected_idempotency = contract_digest(
        canonical_contract_bytes(
            {
                "command_id": command["command_id"],
                "command_hash": command["command_hash"],
                "command_plan_hash": command["command_plan_hash"],
                "attempt_id": command["attempt_id"],
                "generation": command["lifecycle_generation"],
                "phase_execution_id": command["phase_execution_id"],
            }
        )
    )
    if command["idempotency_key"] != expected_idempotency:
        raise IntegrityError("PhaseCommand idempotency_key mismatch")


def _validate_phase_run_receipt(project_root: Path, record: dict[str, Any], receipt_ref: dict[str, Any]) -> dict[str, Any]:
    try:
        return validate_phase_run_receipt(project_root, record, receipt_ref)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise IntegrityError(f"PhaseRunReceipt rejected: {exc}") from exc


def _phase_command(state: dict[str, Any], command_id: str) -> dict[str, Any]:
    record = state["phase_commands"].get(command_id)
    if not isinstance(record, dict):
        raise IntegrityError(f"unknown phase command {command_id}")
    return record


def _validate_command_replay(record: dict[str, Any], command: dict[str, Any]) -> None:
    if canonical_json(record["command"]) != canonical_json(command):
        raise IntegrityError("phase command replay identity conflict")


def _command_unknown_replay_key(command: dict[str, Any]) -> str:
    return "phase-command-unknown:" + canonical_hash({
        "command_id": command["command_id"],
        "command_hash": command["command_hash"],
        "attempt_id": command["attempt_id"],
        "lifecycle_generation": command["lifecycle_generation"],
    })


def _phase_command_operation_result(record: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(record)
    result["event_id"] = event["event_id"]
    result["event_hash"] = event["event_hash"]
    result["sequence"] = event["sequence"]
    return result


def _validate_state_invariants(state: dict[str, Any]) -> None:
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise BreakingSchemaError("unsupported research state schema")
    if len(_active_standard_attempts(state)) > 1:
        raise IntegrityError("project execution_width=1 violated")
    for semantic, direction in state["directions"].items():
        budget = direction["budget"]
        if budget["target"] != TARGET_OUTCOMES or budget["consumed"] < 0 or budget["reserved"] < 0 or budget["consumed"] + budget["reserved"] > TARGET_OUTCOMES:
            raise IntegrityError(f"invalid direction budget for {semantic}: {budget}")
        actual_reserved = sum(1 for attempt in state["attempts"].values() if attempt["direction_semantic_hash"] == semantic and attempt["reserved_slot"])
        if actual_reserved != budget["reserved"]:
            raise IntegrityError("reserved budget does not match attempt reservations")
        actual_consumed = sum(1 for item in state["method_tried_history"] if item["direction_semantic_hash"] == semantic and item.get("consumes_direction_budget"))
        if actual_consumed != budget["consumed"]:
            raise IntegrityError("consumed budget does not match verified outcomes")
        active_standard = [attempt for attempt in state["attempts"].values() if attempt["profile"] == "standard" and attempt["direction_semantic_hash"] == semantic and attempt["reserved_slot"] and attempt["state"] in ACTIVE_ATTEMPT_STATES]
        if len(active_standard) > 1:
            raise IntegrityError("execution_width=1 invariant violated")
        if direction["status"] in {"FINISHED", "EXHAUSTED"} and active_standard:
            raise IntegrityError("closed direction cannot retain an active standard attempt")
    semantic_outcomes = [item["variant_semantic_hash"] for item in state["method_tried_history"] if item.get("consumes_direction_budget")]
    if len(semantic_outcomes) != len(set(semantic_outcomes)):
        raise IntegrityError("duplicate method-evaluable variant semantic hash")
    for command_id, record in state["phase_commands"].items():
        if record["command"]["command_id"] != command_id:
            raise IntegrityError("phase command projection key mismatch")
        if record["status"] not in {"started", "completed", "unknown"}:
            raise IntegrityError("phase command status is invalid")
        if record["status"] == "completed":
            if not isinstance(record.get("receipt_ref"), dict) or not record.get("completed_event_id"):
                raise IntegrityError("completed phase command lacks receipt/event identity")
            project_root = state.get("project_root")
            if project_root:
                _validate_phase_run_receipt(Path(project_root), record, record["receipt_ref"])
        elif record.get("receipt_ref") is not None or record.get("completed_event_id") is not None:
            raise IntegrityError("non-completed phase command carries completion data")
        if record["status"] == "unknown" and (not record.get("unknown_event_id") or not record.get("unknown_reason")):
            raise IntegrityError("unknown phase command lacks event identity/reason")
    for attempt_id, trial in state["trial_results"].items():
        attempt = state["attempts"].get(attempt_id)
        if not attempt or attempt["state"] != "METHOD_COMPLETED" or attempt["phases"].get(trial["completeness"]) != "COMPLETED":
            raise IntegrityError("METHOD_COMPLETED phase/completeness invariant violated")
        candidate_phases = {item["phase"] for item in trial["observations"] if item["role"] == "candidate"}
        if candidate_phases != {trial["completeness"]}:
            raise IntegrityError("TrialResult observations disagree with completeness")
    for attempt in state["attempts"].values():
        validate_contract(attempt, "attempt_record_v8.schema.json")
        _validate_frozen_trial_spec(attempt)
        expected_consumes, expected_reserved_at_reservation = _profile_budget_mapping(attempt["profile"], attempt["attempt_kind"])
        if attempt["consumes_direction_budget"] != expected_consumes:
            raise IntegrityError("attempt consumes_direction_budget violates profile/attempt kind mapping")
        if not expected_reserved_at_reservation and attempt["reserved_slot"]:
            raise IntegrityError("bootstrap attempt cannot hold a direction reservation")
        if attempt["profile"] == "standard" and attempt["state"] in ACTIVE_ATTEMPT_STATES and not attempt["reserved_slot"]:
            raise IntegrityError("active standard attempt must retain its reservation")
        if not isinstance(attempt["lifecycle_generation"], int) or attempt["lifecycle_generation"] < 0:
            raise IntegrityError("attempt lifecycle_generation is invalid")
        if attempt["state"] == "PROXY_RUNNING" and attempt["phases"]["proxy"] != "RUNNING":
            raise IntegrityError("PROXY_RUNNING phase invariant violated")
        if attempt["state"] == "PROXY_COMPLETED" and attempt["phases"]["proxy"] != "COMPLETED":
            raise IntegrityError("PROXY_COMPLETED phase invariant violated")
        if attempt["state"] == "FULL_RUNNING" and attempt["phases"]["full"] != "RUNNING":
            raise IntegrityError("FULL_RUNNING phase invariant violated")
        if attempt["state"] == "READY" and any(value == "RUNNING" for value in attempt["phases"].values()):
            raise IntegrityError("READY attempt cannot retain a RUNNING phase")
        if attempt["state"] == "RESOURCE_PAUSED" and attempt.get("paused_phase") not in {"proxy", "full"}:
            raise IntegrityError("RESOURCE_PAUSED attempt must record paused_phase")
        if attempt["state"] == "RESOURCE_PAUSED" and attempt["phases"][attempt["paused_phase"]] != "RUNNING":
            raise IntegrityError("RESOURCE_PAUSED attempt must preserve its interrupted running phase")
        if attempt["state"] != "RESOURCE_PAUSED" and attempt.get("paused_phase") is not None:
            raise IntegrityError("paused_phase is only valid for RESOURCE_PAUSED")
        if attempt["attempt_kind"] == "proxy_full" and attempt["state"] == "FULL_RUNNING" and "proxy" in attempt["required_phases"] and attempt["phases"]["proxy"] != "COMPLETED":
            raise IntegrityError("proxy_full phase prerequisite invariant violated")
        if attempt["attempt_kind"] == "proxy_full" and attempt["state"] == "PROXY_COMPLETED":
            proxy = attempt.get("committed_proxy_outcome")
            if attempt["phases"]["proxy"] != "COMPLETED" or not isinstance(proxy, dict) or proxy.get("decision") != "RUN_FULL":
                raise IntegrityError("PROXY_COMPLETED requires retained RUN_FULL proxy authorization")
        for phase in ("proxy", "full"):
            execution = attempt["phase_executions"][phase]
            if attempt["phases"][phase] == "RUNNING" and not isinstance(execution, dict):
                raise IntegrityError("running phase requires PhaseExecutionManifest")
        if attempt["state"] == "PROXY_COMPLETED" and not isinstance(attempt.get("committed_proxy_outcome"), dict):
            raise IntegrityError("PROXY_COMPLETED requires committed ProxyOutcome")


def _apply_transition(attempt: dict[str, Any], new_state: str, phase: str | None, phase_state: str | None, timestamp: str) -> None:
    current = attempt["state"]
    if current in TERMINAL_ATTEMPT_STATES or new_state not in TRANSITIONS.get(current, set()):
        raise IntegrityError(f"illegal attempt transition {current} -> {new_state}")
    if (phase is None) != (phase_state is None):
        raise IntegrityError("phase and phase_state must be provided together")
    if phase is not None:
        if phase not in {"proxy", "full"} or phase_state not in {"PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED"}:
            raise IntegrityError("invalid phase transition payload")
        current_phase = attempt["phases"][phase]
        allowed_phase_edges = {
            "PENDING": {"RUNNING", "SKIPPED"},
            "RUNNING": {"COMPLETED", "FAILED"},
            "COMPLETED": set(),
            "FAILED": set(),
            "SKIPPED": set(),
        }
        if phase_state != current_phase and phase_state not in allowed_phase_edges[current_phase]:
            raise IntegrityError(f"phase transition must be monotonic: {phase} {current_phase} -> {phase_state}")
        attempt["phases"][phase] = phase_state
    if new_state == "PROXY_RUNNING" and attempt["phases"]["proxy"] != "RUNNING":
        raise IntegrityError("PROXY_RUNNING requires proxy phase RUNNING")
    if new_state == "PROXY_COMPLETED" and attempt["phases"]["proxy"] != "COMPLETED":
        raise IntegrityError("PROXY_COMPLETED requires proxy phase COMPLETED")
    if new_state == "FULL_RUNNING":
        if attempt["phases"]["full"] != "RUNNING":
            raise IntegrityError("FULL_RUNNING requires full phase RUNNING")
        if attempt["attempt_kind"] == "proxy_full" and "proxy" in attempt["required_phases"] and attempt["phases"]["proxy"] != "COMPLETED":
            raise IntegrityError("proxy_full requires proxy COMPLETED before full execution")
    attempt["state"] = new_state
    attempt["updated_at"] = timestamp


def _active_direction(state: dict[str, Any], semantic: str) -> dict[str, Any]:
    direction = state["directions"].get(semantic)
    if not direction or direction["status"] != "ACTIVE":
        raise IntegrityError("direction is not active")
    return direction


def _attempt(state: dict[str, Any], attempt_id: str) -> dict[str, Any]:
    attempt = state["attempts"].get(attempt_id)
    if not attempt:
        raise IntegrityError(f"unknown attempt_id {attempt_id}")
    return attempt


def _failure_route(failure_class: str) -> tuple[str, str]:
    try:
        return FAILURE_ROUTES[failure_class]
    except KeyError as error:
        raise IntegrityError(f"unknown failure_class {failure_class}") from error


def _active_standard_attempts(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        attempt for attempt in state["attempts"].values()
        if attempt["state"] in ACTIVE_ATTEMPT_STATES
    ]


def _profile_budget_mapping(profile: str, attempt_kind: str) -> tuple[bool, bool]:
    if profile == "bootstrap" and attempt_kind == "bootstrap_proxy":
        return False, False
    if profile == "standard" and attempt_kind in {"proxy", "full", "proxy_full"}:
        return True, True
    raise IntegrityError(f"invalid profile/attempt kind combination: {profile}/{attempt_kind}")


def _operation_event_id(prefix: str, attempt: dict[str, Any], payload: dict[str, Any]) -> str:
    identity = {
        "attempt_id": attempt["attempt_id"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "expected_state": payload.get("expected_state", attempt["state"]),
        "operation": payload,
    }
    return f"{prefix}:{attempt['attempt_id']}:{canonical_hash(identity)}"


def _disposition_replay_key(attempt: dict[str, Any], evidence: dict[str, Any]) -> str:
    return "disposition:" + canonical_hash({"attempt_id": attempt["attempt_id"], "lifecycle_generation": attempt["lifecycle_generation"], "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"], "evidence": evidence})


def _disposition_replay_key_from_evidence(evidence: dict[str, Any]) -> str:
    return "disposition:" + canonical_hash({"attempt_id": evidence["attempt_id"], "lifecycle_generation": evidence["lifecycle_generation"], "implementation_hash": evidence["implementation_hash"], "attempt_input_hash": evidence["attempt_input_hash"], "evidence": evidence})


def _resume_replay_key(evidence: dict[str, Any]) -> str:
    return "resume:" + canonical_hash(evidence)


def _finalization_replay_key(trial_result: dict[str, Any]) -> str:
    return "finalization:" + canonical_hash(trial_result)


def _finalization_attempt_key(attempt_id: str) -> str:
    return f"finalization-attempt:{attempt_id}"


def _transition_replay_key(attempt: dict[str, Any], new_state: str, phase: str | None, phase_state: str | None) -> str:
    return "transition:" + canonical_hash({"attempt_id": attempt["attempt_id"], "lifecycle_generation": attempt["lifecycle_generation"], "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"], "new_state": new_state, "phase": phase, "phase_state": phase_state})


def _transition_replay_key_from_payload(payload: dict[str, Any]) -> str:
    return "transition:" + canonical_hash({key: payload[key] for key in ["attempt_id", "lifecycle_generation", "implementation_hash", "attempt_input_hash", "new_state", "phase", "phase_state"]})


def _revision_replay_key(attempt: dict[str, Any], implementation_hash: str, attempt_input_hash: str) -> str:
    return "revision:" + canonical_hash({
        "attempt_id": attempt["attempt_id"],
        "source_generation": attempt["lifecycle_generation"],
        "source_state": attempt["state"],
        "previous_implementation_hash": attempt["implementation_hash"],
        "previous_attempt_input_hash": attempt["attempt_input_hash"],
        "implementation_hash": implementation_hash,
        "attempt_input_hash": attempt_input_hash,
    })


def _revision_replay_key_from_payload(payload: dict[str, Any]) -> str:
    return "revision:" + canonical_hash({
        "attempt_id": payload["attempt_id"],
        "source_generation": payload["lifecycle_generation"],
        "source_state": payload["expected_state"],
        "previous_implementation_hash": payload["previous_implementation_hash"],
        "previous_attempt_input_hash": payload["previous_attempt_input_hash"],
        "implementation_hash": payload["implementation_hash"],
        "attempt_input_hash": payload["attempt_input_hash"],
    })


def _revision_payload(
    attempt: dict[str, Any],
    implementation_hash: str,
    attempt_input_hash: str,
) -> dict[str, Any]:
    return {
        "attempt_id": attempt["attempt_id"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "previous_implementation_hash": attempt["implementation_hash"],
        "previous_attempt_input_hash": attempt["attempt_input_hash"],
        "expected_state": attempt["state"],
        "implementation_hash": implementation_hash,
        "attempt_input_hash": attempt_input_hash,
    }


def _attempt_input_hash_for_implementation(attempt: dict[str, Any], implementation_hash: str) -> str:
    trial_spec = attempt["frozen_trial_spec"]
    return attempt_input_hash(
        implementation_hash_value=implementation_hash,
        protocol=trial_spec["protocol"],
        sample_manifest=trial_spec["sample_manifest"],
        seeds=trial_spec["statistical_testing"]["seeds"],
        runtime_config=trial_spec["execution_contract"]["runtime_config"],
        evaluator_hash=trial_spec["execution_contract"]["evaluator_hash"],
        trial_spec=trial_spec,
    )


def _validate_frozen_trial_spec(attempt: dict[str, Any]) -> None:
    trial_spec = attempt["frozen_trial_spec"]
    validate_trial_spec(trial_spec)
    expected = {
        "trial_spec_hash": trial_spec_hash(trial_spec),
        "protocol_hash": canonical_hash(trial_spec["protocol"]),
        "sample_manifest_hash": canonical_hash(trial_spec["sample_manifest"]),
        "acceptance_contract_hash": acceptance_contract_hash(trial_spec),
        "runtime_config_hash": canonical_hash(trial_spec["execution_contract"]["runtime_config"]),
        "evaluator_hash": trial_spec["execution_contract"]["evaluator_hash"],
    }
    for key, value in expected.items():
        if attempt[key] != value:
            raise IntegrityError(f"frozen TrialSpec {key} mismatch")
    if attempt["attempt_input_hash"] != _attempt_input_hash_for_implementation(attempt, attempt["implementation_hash"]):
        raise IntegrityError("attempt_input_hash does not bind frozen TrialSpec")


def _validate_trial_spec_projection(project_root: Path, attempt: dict[str, Any]) -> None:
    path = _attempt_trial_spec_projection_path(project_root, attempt)
    if not path.exists():
        raise IntegrityError("canonical TrialSpec projection is missing")
    try:
        projected = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError("TrialSpec projection is corrupt") from error
    if canonical_hash(projected) != attempt["trial_spec_hash"] or projected != attempt["frozen_trial_spec"]:
        raise IntegrityError("TrialSpec projection drift detected")


def _attempt_trial_spec_projection_path(project_root: Path, attempt: dict[str, Any]) -> Path:
    return project_root / "plan" / "attempts" / attempt["attempt_id"] / "trial_spec" / f"{attempt['trial_spec_hash']}.json"


def _artifact_path(project_root: Path, artifact: dict[str, Any]) -> Path:
    path = Path(artifact["path"])
    return path if path.is_absolute() else project_root / path


def _validate_failure_identity(attempt: dict[str, Any], evidence: dict[str, Any]) -> None:
    for key in [
        "attempt_id", "direction_semantic_hash", "direction_spec_hash", "variant_semantic_hash",
        "variant_spec_hash", "trial_spec_hash", "protocol_hash", "sample_manifest_hash",
        "evaluator_hash", "lifecycle_generation", "implementation_hash", "attempt_input_hash",
    ]:
        if evidence[key] != attempt[key]:
            raise IntegrityError(f"FailureEvidence {key} mismatch")
    if evidence["source_state"] != attempt["state"]:
        raise IntegrityError("FailureEvidence source state mismatch")
    expected_phases = {
        "IMPLEMENTING": {"implementation"},
        "READY": {"activation"},
        "PROXY_RUNNING": {"proxy", "activation"},
        "PROXY_COMPLETED": {"activation"},
        "FULL_RUNNING": {"full", "activation"},
    }.get(attempt["state"], set())
    if evidence["source_phase"] not in expected_phases:
        raise IntegrityError("FailureEvidence source phase mismatch")
    if evidence["phase"] != evidence["source_phase"]:
        raise IntegrityError("FailureEvidence phase must equal source_phase")


def _validate_failure_evidence(
    project_root: Path,
    state: dict[str, Any],
    attempt: dict[str, Any],
    evidence: dict[str, Any],
    *,
    allow_core_resource: bool = False,
) -> None:
    _validate_failure_identity(attempt, evidence)
    execution_phase = evidence["source_phase"] if evidence["source_phase"] in {"proxy", "full"} else (
        "proxy" if attempt["state"] == "PROXY_RUNNING" else "full" if attempt["state"] == "FULL_RUNNING" else None
    )
    if execution_phase is not None:
        execution = attempt["phase_executions"].get(execution_phase)
        if not isinstance(execution, dict):
            raise IntegrityError("FailureEvidence requires an authoritative phase execution")
        for key in ("phase_execution_id", "phase_start_event_id"):
            if evidence[key] != execution[key]:
                raise IntegrityError(f"FailureEvidence {key} mismatch")
        if evidence["producer_run_id"] != execution["producer_run_id"]:
            raise IntegrityError("FailureEvidence producer_run_id mismatch")
    _failure_route(evidence["failure_class"])
    authoritative_receipt = _validate_operation_command_binding(
        project_root,
        state,
        attempt,
        evidence,
        allow_core_probe=allow_core_resource and evidence["failure_class"] in {"resource_pause", "oom_retry"},
    )
    _validate_trial_spec_projection(project_root, attempt)
    failure_raw = _read_canonical_operation_evidence(
        project_root,
        attempt,
        evidence["producer_run_id"],
        "failure_evidence",
        evidence_bytes_hash(canonical_evidence_bytes(evidence)),
    )
    expected_identity = _failure_validation_identity(attempt, evidence)
    try:
        if evidence["failure_class"] in {"resource_pause", "oom_retry"}:
            probe_raw = _read_canonical_operation_evidence(
                project_root,
                attempt,
                evidence["producer_run_id"],
                "resource_probe",
                evidence["cross_references"]["resource_probe_hash"],
            )
            validate_failure_evidence_bytes(expected_identity, failure_raw, resource_probe_raw=probe_raw)
        else:
            if authoritative_receipt is None or int(authoritative_receipt["exit_code"]) == 0:
                raise IntegrityError("non-resource failure requires a failed authoritative PhaseRunReceipt")
            if evidence["log_hash"] != authoritative_receipt["stderr_hash"]:
                raise IntegrityError("FailureEvidence log_hash differs from authoritative receipt stderr_hash")
            receipt_raw = canonical_contract_bytes(authoritative_receipt)
            validate_failure_evidence_bytes(
                expected_identity,
                failure_raw,
                phase_run_receipt_raw=receipt_raw,
            )
    except (TypeError, ValueError, KeyError) as exc:
        raise IntegrityError(f"FailureEvidence raw-byte validation failed: {exc}") from exc


def _validate_resume_identity(attempt: dict[str, Any], evidence: dict[str, Any]) -> None:
    for key in [
        "attempt_id", "direction_semantic_hash", "direction_spec_hash", "variant_semantic_hash",
        "variant_spec_hash", "trial_spec_hash", "protocol_hash", "sample_manifest_hash",
        "evaluator_hash", "lifecycle_generation", "implementation_hash", "attempt_input_hash",
    ]:
        if evidence[key] != attempt[key]:
            raise IntegrityError(f"ResumeEvidence {key} mismatch")
    if evidence["phase"] != "resume":
        raise IntegrityError("ResumeEvidence phase mismatch")


def _validate_operation_command_binding(
    project_root: Path,
    state: dict[str, Any],
    attempt: dict[str, Any],
    evidence: dict[str, Any],
    *,
    allow_core_probe: bool,
) -> dict[str, Any] | None:
    receipt_ref = evidence.get("receipt_ref")
    if not isinstance(receipt_ref, dict) or evidence.get("receipt_hash") != receipt_ref.get("digest"):
        raise IntegrityError("operation evidence receipt binding is malformed")
    if allow_core_probe:
        try:
            receipt = ContractStore(project_root).read_json(
                receipt_ref,
                schema_file="core_resource_probe_receipt_v2.schema.json",
            )
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
            raise IntegrityError(f"core resource probe receipt rejected: {exc}") from exc
        expected_fields = {
            "schema_version": "auto_research_core_resource_probe_receipt_v2",
            "attempt_id": attempt["attempt_id"],
            "direction_semantic_hash": attempt["direction_semantic_hash"],
            "direction_spec_hash": attempt["direction_spec_hash"],
            "variant_semantic_hash": attempt["variant_semantic_hash"],
            "variant_spec_hash": attempt["variant_spec_hash"],
            "trial_spec_hash": attempt["trial_spec_hash"],
            "protocol_hash": attempt["protocol_hash"],
            "sample_manifest_hash": attempt["sample_manifest_hash"],
            "evaluator_hash": attempt["evaluator_hash"],
            "lifecycle_generation": attempt["lifecycle_generation"],
            "implementation_hash": attempt["implementation_hash"],
            "attempt_input_hash": attempt["attempt_input_hash"],
            "phase": evidence["phase"],
            "phase_execution_id": evidence["phase_execution_id"],
            "phase_start_event_id": evidence["phase_start_event_id"],
            "authority_event_id": evidence["phase_start_event_id"],
            "producer_run_id": evidence["producer_run_id"],
        }
        if any(receipt.get(key) != value for key, value in expected_fields.items()):
            raise IntegrityError("core resource probe receipt identity mismatch")
        for key in ("resource_type", "resource_id", "required_capacity", "observed_capacity", "unit", "probe_status", "observed_at"):
            if key in evidence and receipt.get(key) != evidence.get(key):
                raise IntegrityError(f"core resource probe receipt {key} mismatch")
        expected_command_hash = canonical_hash({"command_id": evidence["command_id"], "receipt": receipt_ref["digest"]})
        operation = "resume-resource-probe" if evidence.get("phase") == "resume" else "phase-resource-probe"
        expected_operation = "resume" if operation == "resume-resource-probe" else "pause"
        if receipt["operation"] != expected_operation:
            raise IntegrityError("core resource probe operation mismatch")
        expected_plan_hash = canonical_hash({
            "operation": operation,
            "trial_spec_hash": attempt["trial_spec_hash"],
            **(
                {"paused_phase": attempt.get("paused_phase")}
                if operation == "resume-resource-probe"
                else {
                    "phase": receipt.get("phase"),
                    "required": receipt.get("required_capacity"),
                    "unit": receipt.get("unit"),
                }
            ),
        })
        if evidence["command_hash"] != expected_command_hash or evidence["command_plan_hash"] != expected_plan_hash:
            raise IntegrityError("core resource probe command identity mismatch")
        return deepcopy(receipt)
    record = state.get("phase_commands", {}).get(evidence.get("command_id"))
    if not isinstance(record, dict) or record.get("status") != "completed":
        raise IntegrityError("operation evidence requires a completed authoritative command")
    command = record.get("command") or {}
    for key in ("command_id", "command_hash", "command_plan_hash"):
        if evidence.get(key) != command.get(key):
            raise IntegrityError(f"operation evidence {key} mismatch")
    if record.get("receipt_ref") != receipt_ref:
        raise IntegrityError("operation evidence receipt differs from SQLite command record")
    receipt = _validate_phase_run_receipt(project_root, record, receipt_ref)
    expected = {
        "attempt_id": attempt["attempt_id"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase_execution_id": evidence["phase_execution_id"],
        "phase_start_event_id": evidence["phase_start_event_id"],
        "producer_run_id": evidence["producer_run_id"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise IntegrityError("operation evidence receipt identity mismatch")
    if int(receipt["exit_code"]) != int(evidence.get("exit_code", receipt["exit_code"])):
        raise IntegrityError("operation evidence exit_code differs from receipt")
    return receipt


def _validate_resume_evidence(
    project_root: Path,
    state: dict[str, Any],
    attempt: dict[str, Any],
    evidence: dict[str, Any],
    *,
    allow_core_resource: bool | None = None,
) -> None:
    _validate_resume_identity(attempt, evidence)
    if attempt["state"] != "RESOURCE_PAUSED":
        raise IntegrityError("resume requires RESOURCE_PAUSED")
    pause_operation = _latest_pause_operation(state, attempt)
    if pause_operation is None:
        raise IntegrityError("resume requires a committed resource pause event")
    pause_event_id, pause_evidence = pause_operation
    if evidence["pause_event_id"] != pause_event_id or evidence["pause_evidence_hash"] != evidence_bytes_hash(canonical_evidence_bytes(pause_evidence)):
        raise IntegrityError("resume does not bind the committed pause event")
    if allow_core_resource is None:
        allow_core_resource = str(evidence.get("command_id") or "").startswith("core-resource-resume-")
    _validate_operation_command_binding(
        project_root, state, attempt, evidence, allow_core_probe=allow_core_resource
    )
    _validate_trial_spec_projection(project_root, attempt)
    resume_raw = _read_canonical_operation_evidence(
        project_root,
        attempt,
        evidence["producer_run_id"],
        "resume_evidence",
        evidence_bytes_hash(canonical_evidence_bytes(evidence)),
    )
    probe_raw = _read_canonical_operation_evidence(
        project_root,
        attempt,
        evidence["producer_run_id"],
        "resource_probe",
        evidence["cross_references"]["resource_probe_hash"],
    )
    pause_failure_raw = _read_canonical_operation_evidence(
        project_root,
        attempt,
        pause_evidence["producer_run_id"],
        "failure_evidence",
        evidence["pause_evidence_hash"],
    )
    pause_probe_raw = _read_canonical_operation_evidence(
        project_root,
        attempt,
        pause_evidence["producer_run_id"],
        "resource_probe",
        pause_evidence["cross_references"]["resource_probe_hash"],
    )
    try:
        validate_resume_evidence_bytes(
            _resume_validation_identity(attempt, evidence),
            resume_raw,
            resource_probe_raw=probe_raw,
            expected_pause_identity=_failure_validation_identity(attempt, pause_evidence, historical=True),
            pause_failure_raw=pause_failure_raw,
            pause_resource_probe_raw=pause_probe_raw,
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise IntegrityError(f"ResumeEvidence raw-byte validation failed: {exc}") from exc


def _latest_pause_operation(state: dict[str, Any], attempt: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    matches = []
    for operation in state["operation_events"].values():
        evidence = operation.get("payload", {}).get("failure_evidence")
        if (
            isinstance(evidence, dict)
            and evidence.get("attempt_id") == attempt["attempt_id"]
            and evidence.get("lifecycle_generation") == attempt["lifecycle_generation"]
            and evidence.get("source_phase") == attempt.get("paused_phase")
            and evidence.get("failure_class") in {"resource_pause", "oom_retry"}
        ):
            matches.append((operation["event_id"], evidence))
    return matches[-1] if matches else None


def _read_canonical_operation_evidence(
    project_root: Path,
    attempt: dict[str, Any],
    producer_run_id: str,
    kind: str,
    digest: str,
) -> bytes:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{7,127}", producer_run_id):
        raise IntegrityError(f"{kind} producer_run_id is unsafe")
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise IntegrityError(f"{kind} content hash is invalid")
    if kind not in {"failure_evidence", "resource_probe", "resume_evidence"}:
        raise IntegrityError(f"unsupported operation evidence kind: {kind}")
    relative_path = (
        f"experiment/attempts/{attempt['attempt_id']}/{producer_run_id}/{kind}/{digest}.json"
    )
    try:
        raw = EvidenceStore(project_root).read_staged_source(relative_path)
    except ValueError as exc:
        raise IntegrityError(f"{kind} artifact rejected: {exc}") from exc
    if evidence_bytes_hash(raw) != digest:
        raise IntegrityError(f"{kind} artifact hash drift")
    return raw


def _failure_validation_identity(
    attempt: dict[str, Any],
    evidence: dict[str, Any],
    *,
    historical: bool = False,
) -> dict[str, Any]:
    stable = (
        "attempt_id", "direction_semantic_hash", "direction_spec_hash", "variant_semantic_hash",
        "variant_spec_hash", "trial_spec_hash", "protocol_hash", "sample_manifest_hash", "evaluator_hash",
        "lifecycle_generation", "implementation_hash", "attempt_input_hash",
    )
    identity = {key: evidence[key] if historical else attempt[key] for key in stable}
    identity.update(
        {
            "phase": evidence["phase"],
            "phase_execution_id": evidence["phase_execution_id"],
            "phase_start_event_id": evidence["phase_start_event_id"],
            "producer_run_id": evidence["producer_run_id"],
        }
    )
    return identity


def _resume_validation_identity(attempt: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    identity = {
        key: attempt[key]
        for key in (
            "attempt_id", "direction_semantic_hash", "direction_spec_hash", "variant_semantic_hash",
            "variant_spec_hash", "trial_spec_hash", "protocol_hash", "sample_manifest_hash", "evaluator_hash",
            "lifecycle_generation", "implementation_hash", "attempt_input_hash",
        )
    }
    identity.update(
        {
            "phase": "resume",
            "phase_execution_id": evidence["phase_execution_id"],
            "phase_start_event_id": evidence["phase_start_event_id"],
            "producer_run_id": evidence["producer_run_id"],
        }
    )
    return identity


def _apply_failure_disposition(state: dict[str, Any], attempt: dict[str, Any], target_state: str, evidence: dict[str, Any], timestamp: str) -> None:
    if attempt["state"] in TERMINAL_ATTEMPT_STATES or attempt["state"] in {"IMPLEMENTATION_REPAIR", "RESOURCE_PAUSED"}:
        raise IntegrityError(f"attempt state {attempt['state']} cannot be dispositioned")
    source_phase = evidence["source_phase"]
    execution_phase = source_phase if source_phase in {"proxy", "full"} else "proxy" if attempt["state"] == "PROXY_RUNNING" else "full" if attempt["state"] == "FULL_RUNNING" else None
    if execution_phase and target_state != "RESOURCE_PAUSED" and attempt["phases"][execution_phase] == "RUNNING":
        attempt["phases"][execution_phase] = "FAILED"
    attempt["paused_phase"] = execution_phase if target_state == "RESOURCE_PAUSED" else None
    attempt["state"] = target_state
    attempt["failure_class"] = evidence["failure_class"]
    attempt["artifact_hashes"] = {evidence["evidence_id"]: evidence["log_hash"]}
    attempt["updated_at"] = timestamp
    if target_state == "INTEGRITY_BLOCKED" and attempt["reserved_slot"]:
        state["directions"][attempt["direction_semantic_hash"]]["budget"]["reserved"] -= 1
        attempt["reserved_slot"] = False


def _validate_operation_identity(attempt: dict[str, Any], payload: dict[str, Any], *, require_input: bool = True) -> None:
    if payload["lifecycle_generation"] != attempt["lifecycle_generation"]:
        raise IntegrityError("attempt lifecycle_generation mismatch")
    expected_implementation = payload.get("implementation_hash") if require_input else payload.get("previous_implementation_hash")
    if expected_implementation != attempt["implementation_hash"]:
        raise IntegrityError("attempt implementation revision mismatch")
    expected_input = payload.get("attempt_input_hash") if require_input else payload.get("previous_attempt_input_hash")
    if expected_input != attempt["attempt_input_hash"]:
        raise IntegrityError("attempt input revision mismatch")


def _validate_route_for_attempt(route: dict[str, Any], attempt: dict[str, Any]) -> None:
    validate_contract(route, "route_outcome_v4.schema.json")
    for key in ["direction_id", "direction_semantic_hash", "direction_spec_hash", "variant_id", "variant_semantic_hash", "variant_spec_hash", "attempt_id"]:
        if route["identity"][key] != attempt[key]:
            raise IntegrityError(f"RouteOutcome {key} mismatch")


def _validate_completion_request_shape(completion_evidence: dict[str, Any]) -> None:
    if not isinstance(completion_evidence, dict):
        raise IntegrityError("CompletionEvidence must be an object")
    try:
        validate_contract(completion_evidence, "completion_evidence_v3.schema.json")
    except ValueError as exc:
        raise IntegrityError(f"CompletionEvidence v3 rejected: {exc}") from exc


def _classify_completion_evidence(
    *,
    project_root: Path,
    attempt: dict[str, Any],
    completion_evidence: dict[str, Any],
    phase_commands: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    if completion_evidence["attempt_id"] != attempt["attempt_id"]:
        raise IntegrityError("CompletionEvidence attempt identity mismatch")
    if completion_evidence["trial_spec_hash"] != attempt["trial_spec_hash"]:
        raise IntegrityError("CompletionEvidence TrialSpec identity mismatch")
    for key in ("lifecycle_generation", "implementation_hash", "attempt_input_hash"):
        if completion_evidence[key] != attempt[key]:
            raise IntegrityError(f"CompletionEvidence {key} mismatch")
    evidence_bytes: dict[str, bytes] = {}
    manifest_entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    store = EvidenceStore(project_root)
    for entry in completion_evidence["entries"]:
        if not isinstance(entry, dict):
            raise IntegrityError("CompletionEvidence inventory entry must be an object")
        required = {
            "evidence_id", "kind", "relative_path", "content_hash", "schema_version",
            "attempt_id", "producer_run_id", "direction_semantic_hash", "direction_spec_hash",
            "variant_semantic_hash", "variant_spec_hash", "trial_spec_hash", "protocol_hash",
            "sample_manifest_hash", "evaluator_hash", "lifecycle_generation", "implementation_hash",
            "attempt_input_hash", "phase", "phase_execution_id", "phase_start_event_id",
        }
        if set(entry) != required:
            raise IntegrityError("CompletionEvidence inventory entry fields are invalid")
        evidence_id = entry["evidence_id"]
        if evidence_id in seen_ids:
            raise IntegrityError("CompletionEvidence contains duplicate evidence_id")
        seen_ids.add(evidence_id)
        try:
            raw = store.read_entry(entry, attempt)
        except ValueError as exc:
            raise IntegrityError(f"CompletionEvidence artifact rejected: {exc}") from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IntegrityError("CompletionEvidence artifact is not valid JSON") from exc
        manifest_entries.append({**deepcopy(entry), "cross_references": deepcopy(payload.get("cross_references", {}))})
        evidence_bytes[evidence_id] = raw
    manifest = manifest_from_completion_evidence(attempt=attempt, completion_evidence=completion_evidence)
    phases = {entry["phase"] for entry in manifest_entries}
    executions = {entry["phase_execution_id"] for entry in manifest_entries}
    producers = {entry["producer_run_id"] for entry in manifest_entries}
    if len(phases) != 1 or len(executions) != 1 or len(producers) != 1:
        raise IntegrityError("CompletionEvidence cannot mix phase executions or producers")
    phase = next(iter(phases))
    registered = {
        item["kind"]
        for item in attempt["frozen_trial_spec"]["evidence_requirements"]
        if phase in item["applicable_phases"] or "always" in item["applicable_phases"]
    }
    kinds = [entry["kind"] for entry in manifest_entries]
    if len(kinds) != len(set(kinds)):
        raise IntegrityError("CompletionEvidence contains duplicate evidence kind")
    if not set(kinds).issubset(registered):
        raise IntegrityError("CompletionEvidence contains unregistered evidence kind")
    required = {
        item["kind"]
        for item in attempt["frozen_trial_spec"]["evidence_requirements"]
        if item["required"] and (phase in item["applicable_phases"] or "always" in item["applicable_phases"])
    }
    if not required.issubset(kinds):
        raise IntegrityError("CompletionEvidence is missing required phase evidence")
    phase_execution = attempt["phase_executions"].get(phase)
    if not isinstance(phase_execution, dict):
        raise IntegrityError("CompletionEvidence phase has not been authoritatively started")
    for entry in manifest_entries:
        for key in ("phase_execution_id", "phase_start_event_id", "producer_run_id", "lifecycle_generation", "implementation_hash", "attempt_input_hash"):
            if entry[key] != phase_execution[key]:
                raise IntegrityError(f"CompletionEvidence {key} does not match current phase execution")
    try:
        bound = validate_receipt_bound_evidence(
            project_root=project_root,
            attempt=attempt,
            trial_spec=attempt["frozen_trial_spec"],
            manifest=manifest,
            phase_commands=phase_commands,
            phase=phase,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise IntegrityError(f"CompletionEvidence receipt lineage failed: {exc}") from exc
    manifest = bound.manifest
    evidence_bytes = bound.evidence_bytes
    manifest_entries = manifest["entries"]
    manifest_hash = canonical_hash(manifest)
    completion_fingerprint = canonical_hash({
        "operation": "complete_attempt",
        "attempt_identity": {
            "attempt_id": attempt["attempt_id"],
            "lifecycle_generation": attempt["lifecycle_generation"],
            "direction_spec_hash": attempt["direction_spec_hash"],
            "variant_spec_hash": attempt["variant_spec_hash"],
            "trial_spec_hash": attempt["trial_spec_hash"],
            "implementation_hash": attempt["implementation_hash"],
            "attempt_input_hash": attempt["attempt_input_hash"],
        },
        "producer_run_ids": sorted({entry["producer_run_id"] for entry in manifest_entries}),
        "evidence_manifest_hash": manifest_hash,
        "artifact_hashes": {
            entry["relative_path"]: entry["content_hash"]
            for entry in sorted(manifest_entries, key=lambda item: item["relative_path"])
        },
    })
    try:
        trial_result = classify_trial_result(
            attempt=attempt,
            trial_spec=attempt["frozen_trial_spec"],
            evidence_manifest=manifest,
            evidence_bytes=evidence_bytes,
            diagnostic_result=completion_evidence.get("diagnostic_trial_result"),
        )
    except (ValueError, OSError) as exc:
        raise IntegrityError(f"CompletionEvidence classification failed: {exc}") from exc
    return trial_result, completion_fingerprint


def _decode_phase_completion(
    *,
    project_root: Path,
    attempt: dict[str, Any],
    completion_evidence: dict[str, Any],
    expected_phase: str,
    phase_commands: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    for key in ("attempt_id", "trial_spec_hash", "lifecycle_generation", "implementation_hash", "attempt_input_hash"):
        if completion_evidence.get(key) != attempt.get(key):
            raise IntegrityError(f"CompletionEvidence {key} mismatch")
    phase_execution = attempt["phase_executions"].get(expected_phase)
    if not isinstance(phase_execution, dict):
        raise IntegrityError(f"{expected_phase} phase has not been started")
    store = EvidenceStore(project_root)
    evidence_bytes: dict[str, bytes] = {}
    entries: list[dict[str, Any]] = []
    for source in completion_evidence["entries"]:
        entry = deepcopy(source)
        required = {
            "evidence_id", "kind", "relative_path", "content_hash", "schema_version", "attempt_id",
            "producer_run_id", "direction_semantic_hash", "direction_spec_hash", "variant_semantic_hash",
            "variant_spec_hash", "trial_spec_hash", "protocol_hash", "sample_manifest_hash", "evaluator_hash",
            "lifecycle_generation", "implementation_hash", "attempt_input_hash", "phase", "phase_execution_id",
            "phase_start_event_id",
        }
        if set(entry) != required:
            raise IntegrityError("CompletionEvidence inventory entry fields are invalid")
        if entry["phase"] != expected_phase:
            raise IntegrityError("CompletionEvidence phase mismatch")
        for key in ("producer_run_id", "phase_execution_id", "phase_start_event_id", "lifecycle_generation", "implementation_hash", "attempt_input_hash"):
            if entry[key] != phase_execution[key]:
                raise IntegrityError(f"CompletionEvidence {key} does not match phase execution")
        try:
            raw = store.read_entry(entry, attempt)
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"CompletionEvidence artifact rejected: {exc}") from exc
        entry["cross_references"] = deepcopy(payload.get("cross_references", {}))
        entries.append(entry)
        evidence_bytes[entry["evidence_id"]] = raw
    kinds = [entry["kind"] for entry in entries]
    if len(kinds) != len(set(kinds)):
        raise IntegrityError("CompletionEvidence contains duplicate evidence kind")
    applicable = [
        requirement for requirement in attempt["frozen_trial_spec"]["evidence_requirements"]
        if expected_phase in requirement["applicable_phases"] or "always" in requirement["applicable_phases"]
    ]
    registered = {item["kind"] for item in applicable}
    required_kinds = {item["kind"] for item in applicable if item["required"]}
    if not set(kinds).issubset(registered) or not required_kinds.issubset(kinds):
        raise IntegrityError("CompletionEvidence phase inventory violates frozen TrialSpec")
    manifest = manifest_from_completion_evidence(attempt=attempt, completion_evidence=completion_evidence)
    try:
        bound = validate_receipt_bound_evidence(
            project_root=project_root,
            attempt=attempt,
            trial_spec=attempt["frozen_trial_spec"],
            manifest=manifest,
            phase_commands=phase_commands,
            phase=expected_phase,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise IntegrityError(f"CompletionEvidence receipt lineage failed: {exc}") from exc
    manifest = bound.manifest
    observations = list(bound.observations)
    fingerprint = canonical_hash({"operation": f"commit_{expected_phase}_evidence", "attempt_id": attempt["attempt_id"], "generation": attempt["lifecycle_generation"], "manifest_hash": canonical_hash(manifest), "artifact_hashes": sorted(entry["content_hash"] for entry in manifest["entries"])})
    return manifest, observations, fingerprint


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _measure_os_resource(resource_type: str, resource_id: str, unit: str) -> float:
    del resource_id
    if resource_type == "gpu_memory":
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return 0.0
        result = subprocess.run(
            [executable, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return 0.0
        values = []
        for line in result.stdout.splitlines():
            try:
                values.append(float(line.strip()) * 1024 * 1024)
            except ValueError:
                continue
        return max(values, default=0.0)
    if resource_type == "system_memory" and unit == "bytes":
        try:
            return float(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError):
            return 0.0
    if resource_type == "disk" and unit == "bytes":
        return float(shutil.disk_usage(".").free)
    return 0.0


def _write_operation_evidence(
    project_root: Path,
    attempt: dict[str, Any],
    producer_run_id: str,
    kind: str,
    payload: dict[str, Any],
) -> str:
    raw = canonical_evidence_bytes(payload)
    digest = evidence_bytes_hash(raw)
    relative_path = content_addressed_evidence_path(
        attempt_id=attempt["attempt_id"],
        producer_run_id=producer_run_id,
        evidence_kind=kind,
        content_hash=digest,
    )
    EvidenceStore(project_root).write_entry(
        {"relative_path": relative_path, "producer_run_id": producer_run_id, "kind": kind, "content_hash": digest},
        attempt,
        raw,
    )
    return digest
