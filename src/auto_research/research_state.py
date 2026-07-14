"""SQLite-WAL authoritative event store and deterministic S1-S3 reducer."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import uuid
import fcntl
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from .domain_contracts import (
    ATTEMPT_SCHEMA_VERSION,
    DIRECTION_AGGREGATE_SCHEMA_VERSION,
    ROUTE_OUTCOME_SCHEMA_VERSION,
    canonical_hash,
    validate_contract,
    validate_direction_identity,
    validate_trial_evidence,
    validate_trial_result,
    validate_variant_identity,
)
from .s3_validation import validate_ledger_trial_precommit
from .utils import ensure_dir, now_utc

EVENT_SCHEMA_VERSION = "auto_research_event_v2"
STATE_SCHEMA_VERSION = "auto_research_state_v2"
EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._/\-]{0,255}$")
ZERO_HASH = "0" * 64
TARGET_OUTCOMES = 5
ACTIVE_ATTEMPT_STATES = {"READY", "IMPLEMENTING", "IMPLEMENTATION_REPAIR", "PROXY_RUNNING", "PROXY_COMPLETED", "FULL_RUNNING", "RESOURCE_PAUSED"}
TERMINAL_ATTEMPT_STATES = {"METHOD_COMPLETED", "METHOD_FAILED", "INTEGRITY_BLOCKED", "ABANDONED"}
EVENT_TYPES = {
    "DirectionSelected",
    "VariantPlanned",
    "AttemptReserved",
    "AttemptTransitioned",
    "AttemptImplementationRevised",
    "AttemptAbandoned",
    "AttemptDispositioned",
    "AttemptFinalized",
    "AuditMarker",
}
TRANSITIONS = {
    "PLANNED": {"IMPLEMENTING", "READY", "INTEGRITY_BLOCKED", "ABANDONED"},
    "IMPLEMENTING": {"READY", "IMPLEMENTATION_REPAIR", "RESOURCE_PAUSED", "INTEGRITY_BLOCKED", "ABANDONED"},
    "IMPLEMENTATION_REPAIR": {"RESOURCE_PAUSED", "INTEGRITY_BLOCKED", "ABANDONED"},
    "READY": {"PROXY_RUNNING", "FULL_RUNNING", "IMPLEMENTATION_REPAIR", "RESOURCE_PAUSED", "INTEGRITY_BLOCKED", "ABANDONED"},
    "PROXY_RUNNING": {"PROXY_COMPLETED", "IMPLEMENTATION_REPAIR", "RESOURCE_PAUSED", "INTEGRITY_BLOCKED", "ABANDONED"},
    "PROXY_COMPLETED": {"FULL_RUNNING", "IMPLEMENTATION_REPAIR", "RESOURCE_PAUSED", "INTEGRITY_BLOCKED", "ABANDONED"},
    "FULL_RUNNING": {"IMPLEMENTATION_REPAIR", "RESOURCE_PAUSED", "INTEGRITY_BLOCKED", "ABANDONED"},
    "RESOURCE_PAUSED": {"READY", "IMPLEMENTATION_REPAIR", "INTEGRITY_BLOCKED", "ABANDONED"},
}


class IntegrityError(RuntimeError):
    """Raised when authoritative history or a requested transition is inconsistent."""


class BreakingSchemaError(IntegrityError):
    """Raised when a v1 workspace is opened by the v2-only event store."""


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
        self.after_commit_hook = after_commit_hook
        self._reject_v1_workspace()
        self._initialize_db()

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

    def _reject_v1_workspace(self) -> None:
        old_dir = self.meta_dir / "research_events"
        if old_dir.exists() and any(old_dir.glob("*.json")) and not self.db_path.exists():
            raise BreakingSchemaError("Event v1 workspace is unsupported; rerun from S1 with the Event v2 store")

    def events(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return self._validated_events(connection)

    def state(self) -> dict[str, Any]:
        with self._connect() as connection:
            state = self._state_in_transaction(connection)
        self._write_projections(state)
        return state

    def rebuild(self) -> dict[str, Any]:
        return self.state()

    def append(self, event_type: str, payload: dict[str, Any], *, event_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if event_type != "AuditMarker":
            raise IntegrityError("public append only permits AuditMarker; use a constrained domain method")
        return self._transact(event_type, payload, event_id or str(uuid.uuid4()))

    def _transact(self, event_type: str, payload: dict[str, Any], event_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        _validate_event_request(event_type, payload, event_id)
        payload_json = canonical_json(payload)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
                if existing is not None:
                    if existing["event_type"] != event_type or existing["payload_json"] != payload_json:
                        raise IntegrityError(f"event_id conflict for {event_id}")
                    events = self._validated_events(connection)
                    state = _reduce_all(self.project_root.name, events)
                    connection.commit()
                    event = _row_event(existing)
                else:
                    events = self._validated_events(connection)
                    state = _reduce_all(self.project_root.name, events)
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
                    next_state = reduce_event(state, event)
                    _validate_state_invariants(next_state)
                    connection.execute(
                        "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (sequence, event_id, event_type, payload_json, previous_hash, event["event_hash"], created_at, EVENT_SCHEMA_VERSION),
                    )
                    connection.commit()
                    state = next_state
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
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                events = self._validated_events(connection)
                state = _reduce_all(self.project_root.name, events)
                event_type, payload, event_id = build(deepcopy(state))
                _validate_event_request(event_type, payload, event_id)
                payload_json = canonical_json(payload)
                existing = connection.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
                if existing is not None:
                    if existing["event_type"] != event_type or existing["payload_json"] != payload_json:
                        raise IntegrityError(f"event_id conflict for {event_id}")
                    event = _row_event(existing)
                    event_state = _reduce_all(self.project_root.name, events[: event["sequence"]])
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
                    event_state = reduce_event(state, event)
                    connection.execute(
                        "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (sequence, event_id, event_type, payload_json, previous_hash, event["event_hash"], created_at, EVENT_SCHEMA_VERSION),
                    )
                    connection.commit()
                    replayed = False
            except Exception:
                connection.rollback()
                raise
        if self.after_commit_hook is not None and not replayed:
            self.after_commit_hook()
        self._write_projections(event_state)
        return event, event_state, replayed

    def select_direction(self, direction: dict[str, Any], *, event_id: str | None = None) -> dict[str, Any]:
        validate_direction_identity(direction)
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            existing = state["directions"].get(direction["direction_semantic_hash"])
            if existing and existing["status"] in {"FINISHED", "EXHAUSTED"}:
                raise IntegrityError(f"closed direction semantic hash cannot be reopened: {direction['direction_semantic_hash']}")
            return "DirectionSelected", {"direction": direction}, event_id or f"direction:{direction['direction_spec_hash']}"

        _, state, _ = self._domain_transact(build)
        return state

    def plan_variant(self, variant: dict[str, Any], *, feedback_from_attempt_ids: list[str] | None = None, event_id: str | None = None) -> dict[str, Any]:
        payload = {"variant": variant, "feedback_from_attempt_ids": feedback_from_attempt_ids or []}
        _, state, _ = self._domain_transact(lambda _: ("VariantPlanned", payload, event_id or f"variant:{variant['variant_spec_hash']}"))
        return state

    def reserve_attempt(
        self,
        *,
        profile: str,
        direction: dict[str, Any],
        variant: dict[str, Any],
        implementation_hash: str,
        attempt_input_hash: str,
        attempt_kind: str,
        protocol_hash: str,
        sample_manifest_hash: str,
        runtime_config_hash: str,
        evaluator_hash: str,
        seeds: list[int],
        required_datasets: list[str] | None = None,
        required_phases: list[str] | None = None,
        terminal_method_phases: list[str] | None = None,
        required_roles: list[str] | None = None,
        require_complete_seed_coverage: bool = True,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        validate_variant_identity(direction, variant)
        consumes, reserved = _profile_budget_mapping(profile, attempt_kind)
        if not required_datasets or not required_phases or not terminal_method_phases or not required_roles:
            raise IntegrityError("attempt reservation requires frozen datasets, phases, terminal phases, and roles")
        if not set(required_phases).issubset({"proxy", "full", "ablation"}):
            raise IntegrityError("attempt required phases are invalid")
        if not set(terminal_method_phases).issubset(set(required_phases)):
            raise IntegrityError("terminal method phases must be pre-registered required phases")
        if not {"baseline", "candidate"}.issubset(set(required_roles)):
            raise IntegrityError("method attempt requires baseline and candidate roles")
        identity = canonical_hash(
            {
                "profile": profile,
                "attempt_kind": attempt_kind,
                "direction_spec_hash": direction["direction_spec_hash"],
                "variant_spec_hash": variant["variant_spec_hash"],
                "attempt_input_hash": attempt_input_hash,
            }
        )
        attempt_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.project_root.resolve()}:{identity}"))
        timestamp = now_utc()
        attempt = {
            "schema_version": ATTEMPT_SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "profile": profile,
            "direction_id": direction["direction_id"],
            "direction_semantic_hash": direction["direction_semantic_hash"],
            "direction_spec_hash": direction["direction_spec_hash"],
            "variant_id": variant["variant_id"],
            "variant_semantic_hash": variant["variant_semantic_hash"],
            "variant_spec_hash": variant["variant_spec_hash"],
            "implementation_hash": implementation_hash,
            "implementation_revisions": [{"previous_implementation_hash": None, "implementation_hash": implementation_hash, "attempt_input_hash": attempt_input_hash, "created_at": timestamp}],
            "attempt_input_hash": attempt_input_hash,
            "protocol_hash": protocol_hash,
            "sample_manifest_hash": sample_manifest_hash,
            "runtime_config_hash": runtime_config_hash,
            "evaluator_hash": evaluator_hash,
            "seeds": list(seeds),
            "required_datasets": sorted(set(required_datasets)),
            "required_phases": list(required_phases),
            "terminal_method_phases": list(terminal_method_phases),
            "required_roles": sorted(set(required_roles)),
            "require_complete_seed_coverage": bool(require_complete_seed_coverage),
            "attempt_kind": attempt_kind,
            "lifecycle_generation": 0,
            "state": "READY",
            "consumes_direction_budget": consumes,
            "reserved_slot": reserved,
            "phases": {"proxy": "PENDING", "full": "PENDING"},
            "method_evaluable": False,
            "terminal_outcome": None,
            "failure_class": None,
            "artifact_hashes": {},
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        validate_contract(attempt, "attempt_record_v2.schema.json")
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            repair_attempt = next((item for item in state["attempts"].values() if item["profile"] == profile and item["attempt_kind"] == attempt_kind and item["direction_spec_hash"] == direction["direction_spec_hash"] and item["variant_spec_hash"] == variant["variant_spec_hash"] and item["state"] in {"IMPLEMENTATION_REPAIR", "RESOURCE_PAUSED"}), None)
            if repair_attempt is not None:
                if repair_attempt["implementation_hash"] == implementation_hash:
                    if repair_attempt["attempt_input_hash"] != attempt_input_hash:
                        raise IntegrityError("attempt input changed without an implementation revision")
                    raise IntegrityError("repair attempt requires a new implementation revision before reservation")
                payload = _revision_payload(repair_attempt, implementation_hash, attempt_input_hash, protocol_hash, sample_manifest_hash, runtime_config_hash, evaluator_hash)
                key = event_id or _operation_event_id("implementation-revision", repair_attempt, payload)
                return "AttemptImplementationRevised", payload, key
            existing = state["attempts"].get(attempt_id)
            if existing is not None:
                return "AttemptReserved", {"attempt": existing}, event_id or f"attempt:{profile}:{attempt_kind}:{identity}"
            return "AttemptReserved", {"attempt": attempt}, event_id or f"attempt:{profile}:{attempt_kind}:{identity}"

        _, state, _ = self._domain_transact(build)
        matching = next(item for item in state["attempts"].values() if item["profile"] == profile and item["attempt_kind"] == attempt_kind and item["direction_spec_hash"] == direction["direction_spec_hash"] and item["variant_spec_hash"] == variant["variant_spec_hash"] and item["state"] in ACTIVE_ATTEMPT_STATES)
        return deepcopy(matching)

    def transition_attempt(self, attempt_id: str, new_state: str, *, phase: str | None = None, phase_state: str | None = None, event_id: str | None = None) -> dict[str, Any]:
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            attempt = _attempt(state, attempt_id)
            replay = state["operation_events"].get(_transition_replay_key(attempt, new_state, phase, phase_state))
            if replay is not None:
                return "AttemptTransitioned", replay["payload"], replay["event_id"]
            payload = {"attempt_id": attempt_id, "lifecycle_generation": attempt["lifecycle_generation"], "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"], "expected_state": attempt["state"], "new_state": new_state, "phase": phase, "phase_state": phase_state}
            return "AttemptTransitioned", payload, event_id or _operation_event_id("transition", attempt, payload)

        _, state, _ = self._domain_transact(build)
        return deepcopy(state["attempts"][attempt_id])

    def revise_implementation(
        self,
        attempt_id: str,
        *,
        implementation_hash: str,
        attempt_input_hash: str,
        protocol_hash: str,
        sample_manifest_hash: str,
        runtime_config_hash: str,
        evaluator_hash: str,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            attempt = _attempt(state, attempt_id)
            replay = state["operation_events"].get(_revision_replay_key(attempt_id, implementation_hash, attempt_input_hash))
            if replay is not None:
                return "AttemptImplementationRevised", replay["payload"], replay["event_id"]
            payload = _revision_payload(attempt, implementation_hash, attempt_input_hash, protocol_hash, sample_manifest_hash, runtime_config_hash, evaluator_hash)
            return "AttemptImplementationRevised", payload, event_id or _operation_event_id("implementation-revision", attempt, payload)

        _, state, _ = self._domain_transact(build)
        return deepcopy(state["attempts"][attempt_id])

    def abandon_attempt(self, attempt_id: str, *, reason: str, event_id: str | None = None) -> dict[str, Any]:
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            attempt = _attempt(state, attempt_id)
            payload = {"attempt_id": attempt_id, "lifecycle_generation": attempt["lifecycle_generation"], "expected_state": attempt["state"], "reason": reason}
            return "AttemptAbandoned", payload, event_id or _operation_event_id("attempt-abandoned", attempt, payload)

        _, state, _ = self._domain_transact(build)
        return deepcopy(state["attempts"][attempt_id])

    def disposition_attempt(self, attempt_id: str, *, failure_class: str, artifact_hashes: dict[str, str] | None = None, event_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            attempt = _attempt(state, attempt_id)
            replay = state["operation_events"].get(_disposition_replay_key(attempt, failure_class, artifact_hashes or {}))
            if replay is not None:
                return "AttemptDispositioned", replay["payload"], replay["event_id"]
            payload = {"attempt_id": attempt_id, "lifecycle_generation": attempt["lifecycle_generation"], "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"], "expected_state": attempt["state"], "failure_class": failure_class, "artifact_hashes": artifact_hashes or {}}
            return "AttemptDispositioned", payload, event_id or _operation_event_id("attempt-disposition", attempt, payload)

        _, event_state, _ = self._domain_transact(build)
        return deepcopy(event_state["attempts"][attempt_id]), deepcopy(event_state["last_route_outcome"])

    def complete_attempt(self, trial_result: dict[str, Any], *, event_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        validate_trial_result(trial_result)
        _validate_trial_artifact_hashes(self.project_root, trial_result)
        def build(state: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
            replay = state["operation_events"].get(_finalization_replay_key(trial_result))
            if replay is not None:
                return "AttemptFinalized", replay["payload"], replay["event_id"]
            validate_ledger_trial_precommit(
                project_root=self.project_root,
                state=state,
                trial_result=trial_result,
            )
            attempt = _validate_trial_against_state(state, trial_result)
            payload = {"trial_result": trial_result, "lifecycle_generation": attempt["lifecycle_generation"], "expected_state": attempt["state"], "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"]}
            return "AttemptFinalized", payload, event_id or _operation_event_id("attempt-finalized", attempt, payload)

        _, event_state, _ = self._domain_transact(build)
        return deepcopy(event_state["attempts"][trial_result["attempt_id"]]), deepcopy(event_state["last_route_outcome"])

    def validate_trial_precommit(self, trial_result: dict[str, Any]) -> None:
        validate_trial_result(trial_result)
        _validate_trial_artifact_hashes(self.project_root, trial_result)
        with self._connect() as connection:
            state = self._state_in_transaction(connection)
        validate_ledger_trial_precommit(
            project_root=self.project_root,
            state=state,
            trial_result=trial_result,
        )
        _validate_trial_against_state(state, trial_result)

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
        _reduce_all(self.project_root.name, events)
        return events

    def _state_in_transaction(self, connection: sqlite3.Connection) -> dict[str, Any]:
        return _reduce_all(self.project_root.name, self._validated_events(connection))

    def _write_projections(self, state: dict[str, Any]) -> None:
        del state
        lock_path = self.meta_dir / ".research_projection.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            with self._connect() as connection:
                latest = self._state_in_transaction(connection)
            if self.snapshot_path.exists():
                try:
                    projected = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    projected = {}
                if int(projected.get("last_sequence", -1)) > latest["last_sequence"]:
                    raise IntegrityError("projection sequence is ahead of authoritative SQLite state")
            self._atomic_json(self.snapshot_path, latest)
            ensure_dir(self.attempts_dir)
            for stale in self.attempts_dir.glob("*.json"):
                stale.unlink()
            for attempt in latest["attempts"].values():
                self._atomic_json(self.attempts_dir / f"{attempt['attempt_id']}.json", attempt)
            self._replace_projection(self.route_path, latest.get("last_route_outcome"))
            self._replace_projection(self.trial_path, latest.get("latest_trial_result"))
            self._replace_projection(self.aggregate_path, latest.get("latest_direction_aggregate"))
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

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
    return {"schema_version": STATE_SCHEMA_VERSION, "project_id": project_id, "last_sequence": 0, "last_event_hash": ZERO_HASH, "directions": {}, "excluded_direction_semantic_hashes": [], "variants": {}, "attempts": {}, "trial_results": {}, "method_tried_history": [], "implementation_history": [], "operation_events": {}, "current_direction_semantic_hash": None, "current_variant_spec_hash": None, "last_route_outcome": None, "latest_trial_result": None, "latest_direction_aggregate": None, "updated_at": None}


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
        if any(a["profile"] == "standard" and a["direction_semantic_hash"] == semantic and a["reserved_slot"] and a["state"] in ACTIVE_ATTEMPT_STATES for a in next_state["attempts"].values()):
            raise IntegrityError("execution_width=1 forbids planning while another standard attempt is active")
        next_state["variants"][variant["variant_spec_hash"]] = variant
        next_state["current_variant_spec_hash"] = variant["variant_spec_hash"]
    elif event_type == "AttemptReserved":
        attempt = payload["attempt"]
        validate_contract(attempt, "attempt_record_v2.schema.json")
        expected_consumes, expected_reserved = _profile_budget_mapping(attempt["profile"], attempt["attempt_kind"])
        if attempt["consumes_direction_budget"] != expected_consumes or attempt["reserved_slot"] != expected_reserved:
            raise IntegrityError("attempt budget flags do not match profile/attempt_kind")
        if attempt["lifecycle_generation"] != 0:
            raise IntegrityError("new attempt lifecycle_generation must be zero")
        direction = _active_direction(next_state, attempt["direction_semantic_hash"])
        variant = next_state["variants"].get(attempt["variant_spec_hash"])
        if not variant or variant["variant_semantic_hash"] != attempt["variant_semantic_hash"]:
            raise IntegrityError("attempt references unknown VariantSpec")
        if attempt["attempt_id"] in next_state["attempts"]:
            raise IntegrityError("duplicate attempt_id")
        if attempt["reserved_slot"]:
            budget = direction["budget"]
            if budget["consumed"] + budget["reserved"] >= TARGET_OUTCOMES:
                raise IntegrityError("direction budget exhausted before reservation")
            if any(a["profile"] == "standard" and a["direction_semantic_hash"] == attempt["direction_semantic_hash"] and a["reserved_slot"] and a["state"] in ACTIVE_ATTEMPT_STATES for a in next_state["attempts"].values()):
                raise IntegrityError("execution_width=1 reservation conflict")
            budget["reserved"] += 1
        next_state["attempts"][attempt["attempt_id"]] = attempt
    elif event_type == "AttemptTransitioned":
        attempt = _attempt(next_state, payload["attempt_id"])
        _validate_operation_identity(attempt, payload)
        if payload["expected_state"] != attempt["state"]:
            raise IntegrityError("attempt transition source state mismatch")
        if attempt["state"] == "RESOURCE_PAUSED" and payload["new_state"] == "READY":
            attempt["lifecycle_generation"] += 1
            attempt["phases"] = {"proxy": "PENDING", "full": "PENDING"}
            attempt["failure_class"] = None
            attempt["artifact_hashes"] = {}
        _apply_transition(attempt, payload["new_state"], payload.get("phase"), payload.get("phase_state"), event["created_at"])
        next_state["operation_events"][_transition_replay_key(attempt, payload["new_state"], payload.get("phase"), payload.get("phase_state"))] = {"event_id": event["event_id"], "payload": deepcopy(payload)}
    elif event_type == "AttemptImplementationRevised":
        attempt = _attempt(next_state, payload["attempt_id"])
        _validate_operation_identity(attempt, payload, require_input=False)
        if payload["expected_state"] != attempt["state"]:
            raise IntegrityError("implementation revision source state mismatch")
        if attempt["state"] not in {"IMPLEMENTATION_REPAIR", "RESOURCE_PAUSED"}:
            raise IntegrityError("implementation revision requires IMPLEMENTATION_REPAIR or RESOURCE_PAUSED")
        for key in ["protocol_hash", "sample_manifest_hash", "runtime_config_hash", "evaluator_hash"]:
            if payload[key] != attempt[key]:
                raise IntegrityError(f"implementation repair cannot change {key}")
        if payload["implementation_hash"] == attempt["implementation_hash"]:
            raise IntegrityError("implementation repair requires a new implementation_hash")
        previous = attempt["implementation_hash"]
        attempt["implementation_hash"] = payload["implementation_hash"]
        attempt["attempt_input_hash"] = payload["attempt_input_hash"]
        attempt["protocol_hash"] = payload["protocol_hash"]
        attempt["sample_manifest_hash"] = payload["sample_manifest_hash"]
        attempt["runtime_config_hash"] = payload["runtime_config_hash"]
        attempt["evaluator_hash"] = payload["evaluator_hash"]
        attempt["implementation_revisions"].append({"previous_implementation_hash": previous, "implementation_hash": payload["implementation_hash"], "attempt_input_hash": payload["attempt_input_hash"], "created_at": event["created_at"]})
        attempt["lifecycle_generation"] += 1
        attempt["state"] = "READY"
        attempt["phases"] = {"proxy": "PENDING", "full": "PENDING"}
        attempt["failure_class"] = None
        attempt["artifact_hashes"] = {}
        attempt["updated_at"] = event["created_at"]
        next_state["implementation_history"].append({"attempt_id": attempt["attempt_id"], "previous_implementation_hash": previous, "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"]})
        next_state["operation_events"][_revision_replay_key(attempt["attempt_id"], attempt["implementation_hash"], attempt["attempt_input_hash"])] = {"event_id": event["event_id"], "payload": deepcopy(payload)}
    elif event_type == "AttemptAbandoned":
        attempt = _attempt(next_state, payload["attempt_id"])
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
        attempt = _attempt(next_state, payload["attempt_id"])
        _validate_operation_identity(attempt, payload)
        if payload["expected_state"] != attempt["state"]:
            raise IntegrityError("attempt disposition source state mismatch")
        action, target_state = _failure_route(attempt, payload["failure_class"])
        _apply_transition(attempt, target_state, None, None, event["created_at"])
        attempt["failure_class"] = payload["failure_class"]
        attempt["artifact_hashes"] = deepcopy(payload["artifact_hashes"])
        route = build_route_outcome(next_state, action, [payload["failure_class"]], attempt, source_event_id=event["event_id"])
        next_state["last_route_outcome"] = route
        next_state["operation_events"][_disposition_replay_key(attempt, payload["failure_class"], payload["artifact_hashes"])] = {"event_id": event["event_id"], "payload": deepcopy(payload)}
    elif event_type == "AttemptFinalized":
        trial = payload["trial_result"]
        attempt = _attempt(next_state, trial["attempt_id"])
        _validate_operation_identity(attempt, payload)
        if payload["expected_state"] != attempt["state"]:
            raise IntegrityError("attempt finalization source state mismatch")
        attempt = _validate_trial_against_state(next_state, trial)
        next_state = _apply_trial_to_state(next_state, trial)
        expected_aggregate = _build_direction_aggregate(next_state, attempt["direction_semantic_hash"]) if next_state["directions"][attempt["direction_semantic_hash"]]["budget"]["consumed"] == TARGET_OUTCOMES else None
        route = _route_after_verified_trial(next_state, attempt, expected_aggregate, source_event_id=event["event_id"])
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
    elif event_type == "AuditMarker":
        pass
    else:
        raise IntegrityError(f"unknown event_type {event_type}")
    next_state["last_sequence"] = event["sequence"]
    next_state["last_event_hash"] = event["event_hash"]
    next_state["updated_at"] = event["created_at"]
    _validate_state_invariants(next_state)
    return next_state


def build_route_outcome(state: dict[str, Any], next_action: str, reason_codes: list[str], attempt: dict[str, Any], *, source_event_id: str | None) -> dict[str, Any]:
    budget = deepcopy(state["directions"][attempt["direction_semantic_hash"]]["budget"])
    route = {"schema_version": ROUTE_OUTCOME_SCHEMA_VERSION, "source": {"event_id": source_event_id, "attempt_id": attempt["attempt_id"]}, "identity": {key: attempt[key] for key in ["direction_id", "direction_semantic_hash", "direction_spec_hash", "variant_id", "variant_semantic_hash", "variant_spec_hash", "attempt_id"]}, "next_action": next_action, "reason_codes": list(reason_codes), "budget_snapshot": budget, "artifact_hashes": deepcopy(attempt.get("artifact_hashes") or {}), "idempotency_key": canonical_hash({"source_event_id": source_event_id, "attempt_id": attempt["attempt_id"], "lifecycle_generation": attempt["lifecycle_generation"], "next_action": next_action, "reason_codes": list(reason_codes), "budget": budget, "artifact_hashes": attempt.get("artifact_hashes") or {}, "variant_spec_hash": attempt["variant_spec_hash"]})}
    validate_contract(route, "route_outcome_v2.schema.json")
    return route


def _route_after_verified_trial(state: dict[str, Any], attempt: dict[str, Any], aggregate: dict[str, Any] | None, *, source_event_id: str) -> dict[str, Any]:
    if attempt["profile"] == "bootstrap":
        if attempt["attempt_kind"] != "bootstrap_proxy":
            raise IntegrityError("bootstrap FINISH_RUN requires bootstrap_proxy attempt kind")
        return build_route_outcome(state, "FINISH_RUN", ["bootstrap_proxy_verified"], state["attempts"][attempt["attempt_id"]], source_event_id=source_event_id)
    budget = state["directions"][attempt["direction_semantic_hash"]]["budget"]
    if budget["consumed"] < TARGET_OUTCOMES:
        return build_route_outcome(state, "PROPOSE_NEXT_VARIANT", ["verified_outcome_recorded", "direction_budget_remaining"], state["attempts"][attempt["attempt_id"]], source_event_id=source_event_id)
    if aggregate is None:
        raise IntegrityError("fifth outcome requires DirectionOutcomeAggregate")
    action = "FINISH_DIRECTION" if any(item["outcome"] == "accepted" for item in aggregate["outcomes"]) else "START_NEW_DIRECTION"
    return build_route_outcome(state, action, ["five_verified_outcomes", aggregate["selection"]["status"]], state["attempts"][attempt["attempt_id"]], source_event_id=source_event_id)


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
    next_state["method_tried_history"].append({"attempt_id": attempt["attempt_id"], "profile": attempt["profile"], "consumes_direction_budget": attempt["consumes_direction_budget"], "direction_semantic_hash": attempt["direction_semantic_hash"], "direction_spec_hash": attempt["direction_spec_hash"], "variant_id": attempt["variant_id"], "variant_semantic_hash": attempt["variant_semantic_hash"], "variant_spec_hash": attempt["variant_spec_hash"], "method_evaluable": True, "outcome_classification": trial["outcome_classification"], "trial_result_hash": canonical_hash(trial), "primary_metric_summary": trial["primary_metric_summary"]})
    return next_state


def _validate_trial_against_state(state: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    validate_trial_result(trial)
    attempt = _attempt(state, trial["attempt_id"])
    if attempt["state"] in TERMINAL_ATTEMPT_STATES or attempt["method_evaluable"]:
        existing = state["trial_results"].get(attempt["attempt_id"])
        if existing == trial:
            return attempt
        raise IntegrityError("attempt already finalized")
    for key in ["direction_id", "direction_semantic_hash", "direction_spec_hash", "variant_id", "variant_semantic_hash", "variant_spec_hash", "attempt_input_hash", "protocol_hash"]:
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
    required = set(attempt["required_datasets"])
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
        if observation["raw_artifact_hash"] not in registered_artifacts:
            raise IntegrityError("observation references an unregistered raw artifact hash")
        identity = (observation["phase"], observation["role"], observation["dataset_id"], observation["metric_id"], observation["seed"])
        if identity in identities:
            raise IntegrityError("duplicate observation identity")
        identities.add(identity)
        observed_seeds.add(observation["seed"])
        observation_datasets.add(observation["dataset_id"])
        roles_by_dataset_seed.setdefault((observation["dataset_id"], observation["seed"]), set()).add(observation["role"])
    if observation_datasets != required:
        raise IntegrityError("observation dataset coverage mismatch")
    if attempt["require_complete_seed_coverage"] and observed_seeds != set(attempt["seeds"]):
        raise IntegrityError("observation seed coverage mismatch")
    for dataset_id in required:
        for seed in attempt["seeds"]:
            if attempt["require_complete_seed_coverage"] and not set(attempt["required_roles"]).issubset(roles_by_dataset_seed.get((dataset_id, seed), set())):
                raise IntegrityError("pre-registered role coverage is incomplete")
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
    required = {"DirectionSelected": {"direction"}, "VariantPlanned": {"variant", "feedback_from_attempt_ids"}, "AttemptReserved": {"attempt"}, "AttemptTransitioned": {"attempt_id", "lifecycle_generation", "implementation_hash", "attempt_input_hash", "expected_state", "new_state", "phase", "phase_state"}, "AttemptImplementationRevised": {"attempt_id", "lifecycle_generation", "previous_implementation_hash", "previous_attempt_input_hash", "expected_state", "implementation_hash", "attempt_input_hash", "protocol_hash", "sample_manifest_hash", "runtime_config_hash", "evaluator_hash"}, "AttemptAbandoned": {"attempt_id", "lifecycle_generation", "expected_state", "reason"}, "AttemptDispositioned": {"attempt_id", "lifecycle_generation", "implementation_hash", "attempt_input_hash", "expected_state", "failure_class", "artifact_hashes"}, "AttemptFinalized": {"trial_result", "lifecycle_generation", "expected_state", "implementation_hash", "attempt_input_hash"}, "AuditMarker": {"index"}}[event_type]
    if set(payload) != required:
        raise IntegrityError(f"{event_type} payload fields must be {sorted(required)}")


def _validate_event(event: dict[str, Any]) -> None:
    if event.get("schema_version") != EVENT_SCHEMA_VERSION:
        raise BreakingSchemaError("invalid or unsupported Event v2 schema")
    try:
        validate_contract(event, "event_v2.schema.json")
    except ValueError as exc:
        raise IntegrityError(f"invalid Event v2 schema: {exc}") from exc
    _validate_event_request(event["event_type"], event["payload"], event["event_id"])


def _event_hash(event: dict[str, Any]) -> str:
    return canonical_hash({key: event[key] for key in ["schema_version", "event_id", "sequence", "event_type", "previous_event_hash", "created_at", "payload"]})


def _row_event(row: sqlite3.Row) -> dict[str, Any]:
    return {"schema_version": row["schema_version"], "event_id": row["event_id"], "sequence": row["sequence"], "event_type": row["event_type"], "previous_event_hash": row["previous_event_hash"], "event_hash": row["event_hash"], "created_at": row["created_at"], "payload": json.loads(row["payload_json"])}


def _reduce_all(project_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    state = initial_state(project_id)
    state["project_root"] = None
    for event in events:
        state = reduce_event(state, event)
    state.pop("project_root", None)
    return state


def _validate_state_invariants(state: dict[str, Any]) -> None:
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
    for attempt_id, trial in state["trial_results"].items():
        attempt = state["attempts"].get(attempt_id)
        if not attempt or attempt["state"] != "METHOD_COMPLETED" or attempt["phases"].get(trial["completeness"]) != "COMPLETED":
            raise IntegrityError("METHOD_COMPLETED phase/completeness invariant violated")
        candidate_phases = {item["phase"] for item in trial["observations"] if item["role"] == "candidate"}
        if candidate_phases != {trial["completeness"]}:
            raise IntegrityError("TrialResult observations disagree with completeness")
    for attempt in state["attempts"].values():
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
        if attempt["attempt_kind"] == "proxy_full" and attempt["state"] == "FULL_RUNNING" and "proxy" in attempt["required_phases"] and attempt["phases"]["proxy"] != "COMPLETED":
            raise IntegrityError("proxy_full phase prerequisite invariant violated")


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


def _failure_route(attempt: dict[str, Any], failure_class: str) -> tuple[str, str]:
    if failure_class in {"resource_paused", "oom_retry", "resource_unavailable"}:
        return "PAUSE_RESOURCE", "RESOURCE_PAUSED"
    if failure_class in {"integrity", "safety", "identity_mismatch", "artifact_hash_mismatch"}:
        return "BLOCK_INTEGRITY", "INTEGRITY_BLOCKED"
    return "REPAIR_IMPLEMENTATION", "IMPLEMENTATION_REPAIR"


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


def _disposition_replay_key(attempt: dict[str, Any], failure_class: str, artifact_hashes: dict[str, str]) -> str:
    return "disposition:" + canonical_hash({"attempt_id": attempt["attempt_id"], "lifecycle_generation": attempt["lifecycle_generation"], "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"], "failure_class": failure_class, "artifact_hashes": artifact_hashes})


def _finalization_replay_key(trial_result: dict[str, Any]) -> str:
    return "finalization:" + canonical_hash(trial_result)


def _transition_replay_key(attempt: dict[str, Any], new_state: str, phase: str | None, phase_state: str | None) -> str:
    return "transition:" + canonical_hash({"attempt_id": attempt["attempt_id"], "lifecycle_generation": attempt["lifecycle_generation"], "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"], "new_state": new_state, "phase": phase, "phase_state": phase_state})


def _revision_replay_key(attempt_id: str, implementation_hash: str, attempt_input_hash: str) -> str:
    return "revision:" + canonical_hash({"attempt_id": attempt_id, "implementation_hash": implementation_hash, "attempt_input_hash": attempt_input_hash})


def _revision_payload(
    attempt: dict[str, Any],
    implementation_hash: str,
    attempt_input_hash: str,
    protocol_hash: str,
    sample_manifest_hash: str,
    runtime_config_hash: str,
    evaluator_hash: str,
) -> dict[str, Any]:
    return {
        "attempt_id": attempt["attempt_id"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "previous_implementation_hash": attempt["implementation_hash"],
        "previous_attempt_input_hash": attempt["attempt_input_hash"],
        "expected_state": attempt["state"],
        "implementation_hash": implementation_hash,
        "attempt_input_hash": attempt_input_hash,
        "protocol_hash": protocol_hash,
        "sample_manifest_hash": sample_manifest_hash,
        "runtime_config_hash": runtime_config_hash,
        "evaluator_hash": evaluator_hash,
    }


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
    validate_contract(route, "route_outcome_v2.schema.json")
    for key in ["direction_id", "direction_semantic_hash", "direction_spec_hash", "variant_id", "variant_semantic_hash", "variant_spec_hash", "attempt_id"]:
        if route["identity"][key] != attempt[key]:
            raise IntegrityError(f"RouteOutcome {key} mismatch")


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_trial_artifact_hashes(project_root: Path, trial: dict[str, Any]) -> None:
    for artifact_path, expected_hash in trial["raw_artifacts"].items():
        path = Path(artifact_path)
        if not path.is_absolute():
            path = project_root / path
        if not path.exists() or not path.is_file():
            raise IntegrityError(f"raw artifact missing: {artifact_path}")
        if _sha256(path) != expected_hash:
            raise IntegrityError(f"raw artifact hash mismatch: {artifact_path}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
