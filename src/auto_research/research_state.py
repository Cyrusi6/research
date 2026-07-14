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
    validate_trial_result,
    validate_variant_identity,
)
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
    "IMPLEMENTATION_REPAIR": {"READY", "RESOURCE_PAUSED", "INTEGRITY_BLOCKED", "ABANDONED"},
    "READY": {"PROXY_RUNNING", "FULL_RUNNING", "IMPLEMENTATION_REPAIR", "RESOURCE_PAUSED", "INTEGRITY_BLOCKED", "ABANDONED"},
    "PROXY_RUNNING": {"PROXY_COMPLETED", "IMPLEMENTATION_REPAIR", "RESOURCE_PAUSED", "INTEGRITY_BLOCKED", "ABANDONED"},
    "PROXY_COMPLETED": {"FULL_RUNNING", "IMPLEMENTATION_REPAIR", "RESOURCE_PAUSED", "INTEGRITY_BLOCKED", "ABANDONED"},
    "FULL_RUNNING": {"IMPLEMENTATION_REPAIR", "RESOURCE_PAUSED", "INTEGRITY_BLOCKED", "ABANDONED"},
    "RESOURCE_PAUSED": {"READY", "PROXY_RUNNING", "FULL_RUNNING", "IMPLEMENTATION_REPAIR", "INTEGRITY_BLOCKED", "ABANDONED"},
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
        event_id = event_id or str(uuid.uuid4())
        event, state = self._transact(event_type, payload, event_id)
        return event, state

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

    def select_direction(self, direction: dict[str, Any], *, event_id: str | None = None) -> dict[str, Any]:
        validate_direction_identity(direction)
        _, state = self._transact("DirectionSelected", {"direction": direction}, event_id or f"direction:{direction['direction_spec_hash']}")
        return state

    def plan_variant(self, variant: dict[str, Any], *, feedback_from_attempt_ids: list[str] | None = None, event_id: str | None = None) -> dict[str, Any]:
        payload = {"variant": variant, "feedback_from_attempt_ids": feedback_from_attempt_ids or []}
        _, state = self._transact("VariantPlanned", payload, event_id or f"variant:{variant['variant_spec_hash']}")
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
        event_id: str | None = None,
    ) -> dict[str, Any]:
        validate_variant_identity(direction, variant)
        current_state = self.state()
        repair_attempt = next(
            (
                item
                for item in current_state.get("attempts", {}).values()
                if item.get("profile") == profile
                and item.get("attempt_kind") == attempt_kind
                and item.get("direction_spec_hash") == direction["direction_spec_hash"]
                and item.get("variant_spec_hash") == variant["variant_spec_hash"]
                and item.get("state") in {"IMPLEMENTATION_REPAIR", "RESOURCE_PAUSED"}
            ),
            None,
        )
        if isinstance(repair_attempt, dict):
            if repair_attempt["implementation_hash"] == implementation_hash:
                if repair_attempt["attempt_input_hash"] != attempt_input_hash:
                    raise IntegrityError("attempt input changed without an implementation revision")
                return deepcopy(repair_attempt)
            return self.revise_implementation(
                repair_attempt["attempt_id"],
                implementation_hash=implementation_hash,
                attempt_input_hash=attempt_input_hash,
                protocol_hash=protocol_hash,
                sample_manifest_hash=sample_manifest_hash,
                runtime_config_hash=runtime_config_hash,
                evaluator_hash=evaluator_hash,
            )
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
        existing = self.state().get("attempts", {}).get(attempt_id)
        if isinstance(existing, dict):
            for key, expected in {
                "profile": profile,
                "attempt_kind": attempt_kind,
                "direction_spec_hash": direction["direction_spec_hash"],
                "variant_spec_hash": variant["variant_spec_hash"],
                "attempt_input_hash": attempt_input_hash,
            }.items():
                if existing.get(key) != expected:
                    raise IntegrityError(f"existing attempt identity conflict for {key}")
            return deepcopy(existing)
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
            "attempt_kind": attempt_kind,
            "state": "READY",
            "consumes_direction_budget": profile == "standard" and attempt_kind != "bootstrap_proxy",
            "reserved_slot": profile == "standard" and attempt_kind != "bootstrap_proxy",
            "phases": {"proxy": "PENDING", "full": "PENDING"},
            "method_evaluable": False,
            "terminal_outcome": None,
            "failure_class": None,
            "artifact_hashes": {},
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        validate_contract(attempt, "attempt_record_v2.schema.json")
        payload = {"attempt": attempt}
        event_key = event_id or f"attempt:{profile}:{attempt_kind}:{identity}"
        _, state = self._transact("AttemptReserved", payload, event_key)
        return deepcopy(state["attempts"][attempt_id])

    def transition_attempt(self, attempt_id: str, new_state: str, *, phase: str | None = None, phase_state: str | None = None, event_id: str | None = None) -> dict[str, Any]:
        payload = {"attempt_id": attempt_id, "new_state": new_state, "phase": phase, "phase_state": phase_state}
        _, state = self._transact("AttemptTransitioned", payload, event_id or f"transition:{attempt_id}:{new_state}:{phase or '-'}:{phase_state or '-'}")
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
        payload = {
            "attempt_id": attempt_id,
            "implementation_hash": implementation_hash,
            "attempt_input_hash": attempt_input_hash,
            "protocol_hash": protocol_hash,
            "sample_manifest_hash": sample_manifest_hash,
            "runtime_config_hash": runtime_config_hash,
            "evaluator_hash": evaluator_hash,
        }
        _, state = self._transact("AttemptImplementationRevised", payload, event_id or f"implementation-revision:{attempt_id}:{implementation_hash}:{attempt_input_hash}")
        return deepcopy(state["attempts"][attempt_id])

    def abandon_attempt(self, attempt_id: str, *, reason: str, event_id: str | None = None) -> dict[str, Any]:
        _, state = self._transact("AttemptAbandoned", {"attempt_id": attempt_id, "reason": reason}, event_id or f"attempt-abandoned:{attempt_id}")
        return deepcopy(state["attempts"][attempt_id])

    def disposition_attempt(self, attempt_id: str, *, failure_class: str, artifact_hashes: dict[str, str] | None = None, event_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        state = self.state()
        attempt = state["attempts"][attempt_id]
        action, target_state = _failure_route(attempt, failure_class)
        event_key = event_id or f"attempt-disposition:{attempt_id}:{failure_class}"
        existing_route = state.get("last_route_outcome")
        if (
            attempt.get("state") == target_state
            and attempt.get("failure_class") == failure_class
            and isinstance(existing_route, dict)
            and (existing_route.get("source") or {}).get("event_id") == event_key
        ):
            return deepcopy(attempt), deepcopy(existing_route)
        route = build_route_outcome(state, action, [failure_class], attempt, source_event_id=event_key)
        payload = {"attempt_id": attempt_id, "failure_class": failure_class, "target_state": target_state, "artifact_hashes": artifact_hashes or {}, "route_outcome": route}
        _, next_state = self._transact("AttemptDispositioned", payload, event_key)
        return deepcopy(next_state["attempts"][attempt_id]), deepcopy(next_state["last_route_outcome"])

    def complete_attempt(self, trial_result: dict[str, Any], *, event_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        validate_trial_result(trial_result)
        _validate_trial_artifact_hashes(self.project_root, trial_result)
        event_key = event_id or f"attempt-finalized:{trial_result['attempt_id']}:{canonical_hash(trial_result)}"
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute("SELECT * FROM events WHERE event_id = ?", (event_key,)).fetchone()
                if existing is not None:
                    existing_payload = json.loads(existing["payload_json"])
                    if existing["event_type"] != "AttemptFinalized" or existing_payload.get("trial_result") != trial_result:
                        raise IntegrityError(f"event_id conflict for {event_key}")
                    state = _reduce_all(self.project_root.name, self._validated_events(connection))
                    connection.commit()
                    self._write_projections(state)
                    return deepcopy(state["attempts"][trial_result["attempt_id"]]), deepcopy(state["last_route_outcome"])
                events = self._validated_events(connection)
                state = _reduce_all(self.project_root.name, events)
                attempt = _validate_trial_against_state(state, trial_result)
                preview = _apply_trial_to_state(state, trial_result)
                aggregate = _build_direction_aggregate(preview, attempt["direction_semantic_hash"]) if preview["directions"][attempt["direction_semantic_hash"]]["budget"]["consumed"] == TARGET_OUTCOMES else None
                route = _route_after_verified_trial(preview, attempt, aggregate, source_event_id=event_key)
                payload = {"trial_result": trial_result, "route_outcome": route, "aggregate": aggregate}
                _validate_event_request("AttemptFinalized", payload, event_key)
                payload_json = canonical_json(payload)
                sequence = len(events) + 1
                previous_hash = events[-1]["event_hash"] if events else ZERO_HASH
                created_at = now_utc()
                event = {"schema_version": EVENT_SCHEMA_VERSION, "event_id": event_key, "sequence": sequence, "event_type": "AttemptFinalized", "previous_event_hash": previous_hash, "created_at": created_at, "payload": payload}
                event["event_hash"] = _event_hash(event)
                state = reduce_event(state, event)
                _validate_state_invariants(state)
                connection.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (sequence, event_key, "AttemptFinalized", payload_json, previous_hash, event["event_hash"], created_at, EVENT_SCHEMA_VERSION))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        if self.after_commit_hook is not None:
            self.after_commit_hook()
        self._write_projections(state)
        return deepcopy(state["attempts"][trial_result["attempt_id"]]), deepcopy(state["last_route_outcome"])

    def validate_trial_precommit(self, trial_result: dict[str, Any]) -> None:
        validate_trial_result(trial_result)
        _validate_trial_artifact_hashes(self.project_root, trial_result)
        with self._connect() as connection:
            state = self._state_in_transaction(connection)
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
        self._atomic_json(self.snapshot_path, state)
        ensure_dir(self.attempts_dir)
        for stale in self.attempts_dir.glob("*.json"):
            stale.unlink()
        for attempt in state["attempts"].values():
            self._atomic_json(self.attempts_dir / f"{attempt['attempt_id']}.json", attempt)
        if state.get("last_route_outcome"):
            self._atomic_json(self.route_path, state["last_route_outcome"])
        elif self.route_path.exists():
            self.route_path.unlink()
        latest_trial = state.get("latest_trial_result")
        if latest_trial:
            self._atomic_json(self.trial_path, latest_trial)
        elif self.trial_path.exists():
            self.trial_path.unlink()
        aggregate = state.get("latest_direction_aggregate")
        if aggregate:
            self._atomic_json(self.aggregate_path, aggregate)
        elif self.aggregate_path.exists():
            self.aggregate_path.unlink()

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
    return {"schema_version": STATE_SCHEMA_VERSION, "project_id": project_id, "last_sequence": 0, "last_event_hash": ZERO_HASH, "directions": {}, "excluded_direction_semantic_hashes": [], "variants": {}, "attempts": {}, "trial_results": {}, "method_tried_history": [], "implementation_history": [], "current_direction_semantic_hash": None, "current_variant_spec_hash": None, "last_route_outcome": None, "latest_trial_result": None, "latest_direction_aggregate": None, "updated_at": None}


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
        _apply_transition(attempt, payload["new_state"], payload.get("phase"), payload.get("phase_state"), event["created_at"])
    elif event_type == "AttemptImplementationRevised":
        attempt = _attempt(next_state, payload["attempt_id"])
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
        attempt["state"] = "READY"
        attempt["updated_at"] = event["created_at"]
        next_state["implementation_history"].append({"attempt_id": attempt["attempt_id"], "previous_implementation_hash": previous, "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"]})
    elif event_type == "AttemptAbandoned":
        attempt = _attempt(next_state, payload["attempt_id"])
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
        _apply_transition(attempt, payload["target_state"], None, None, event["created_at"])
        attempt["failure_class"] = payload["failure_class"]
        attempt["artifact_hashes"] = deepcopy(payload["artifact_hashes"])
        route = payload["route_outcome"]
        _validate_route_for_attempt(route, attempt)
        next_state["last_route_outcome"] = route
    elif event_type == "AttemptFinalized":
        trial = payload["trial_result"]
        attempt = _validate_trial_against_state(next_state, trial)
        next_state = _apply_trial_to_state(next_state, trial)
        route = payload["route_outcome"]
        _validate_route_for_attempt(route, next_state["attempts"][attempt["attempt_id"]])
        expected_aggregate = _build_direction_aggregate(next_state, attempt["direction_semantic_hash"]) if next_state["directions"][attempt["direction_semantic_hash"]]["budget"]["consumed"] == TARGET_OUTCOMES else None
        if payload.get("aggregate") != expected_aggregate:
            raise IntegrityError("DirectionOutcomeAggregate is not deterministic")
        if expected_aggregate:
            next_state["directions"][attempt["direction_semantic_hash"]]["aggregate"] = expected_aggregate
            next_state["directions"][attempt["direction_semantic_hash"]]["status"] = "FINISHED" if route["next_action"] == "FINISH_DIRECTION" else "EXHAUSTED"
            next_state["latest_direction_aggregate"] = expected_aggregate
            next_state["excluded_direction_semantic_hashes"] = sorted(set(next_state["excluded_direction_semantic_hashes"] + [attempt["direction_semantic_hash"]]))
            next_state["current_direction_semantic_hash"] = None
            next_state["current_variant_spec_hash"] = None
        next_state["last_route_outcome"] = route
        next_state["latest_trial_result"] = trial
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
    route = {"schema_version": ROUTE_OUTCOME_SCHEMA_VERSION, "source": {"event_id": source_event_id, "attempt_id": attempt["attempt_id"]}, "identity": {key: attempt[key] for key in ["direction_id", "direction_semantic_hash", "direction_spec_hash", "variant_id", "variant_semantic_hash", "variant_spec_hash", "attempt_id"]}, "next_action": next_action, "reason_codes": list(reason_codes), "budget_snapshot": budget, "artifact_hashes": deepcopy(attempt.get("artifact_hashes") or {}), "idempotency_key": canonical_hash({"attempt_id": attempt["attempt_id"], "next_action": next_action, "budget": budget, "variant_spec_hash": attempt["variant_spec_hash"]})}
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
    if attempt["state"] not in {"PROXY_RUNNING", "PROXY_COMPLETED", "FULL_RUNNING", "READY"}:
        raise IntegrityError(f"attempt state {attempt['state']} cannot finalize")
    if trial["completeness"] == "proxy" and attempt["attempt_kind"] not in {"proxy", "bootstrap_proxy"}:
        raise IntegrityError("proxy cannot be terminal for this attempt kind")
    if trial["completeness"] == "full" and attempt["attempt_kind"] not in {"full", "proxy_full"}:
        raise IntegrityError("full result cannot finalize this attempt kind")
    if attempt["consumes_direction_budget"] and any(item.get("consumes_direction_budget") and item["variant_semantic_hash"] == attempt["variant_semantic_hash"] and item["attempt_id"] != attempt["attempt_id"] for item in state["method_tried_history"]):
        raise IntegrityError("variant semantic hash already has a method-evaluable outcome")
    return attempt


def _build_direction_aggregate(state: dict[str, Any], semantic: str) -> dict[str, Any]:
    outcomes = [item for item in state["method_tried_history"] if item["direction_semantic_hash"] == semantic and item.get("consumes_direction_budget")]
    if len(outcomes) != TARGET_OUTCOMES or len({item["attempt_id"] for item in outcomes}) != TARGET_OUTCOMES or len({item["variant_semantic_hash"] for item in outcomes}) != TARGET_OUTCOMES:
        raise IntegrityError("DirectionOutcomeAggregate requires exactly five unique attempts and semantic variants")
    rows = [{key: item[key] for key in ["attempt_id", "variant_id", "variant_semantic_hash", "variant_spec_hash", "trial_result_hash", "outcome_classification", "primary_metric_summary"]} for item in outcomes]
    for row in rows:
        row["outcome"] = row.pop("outcome_classification")
    comparable = [row for row in rows if isinstance(row["primary_metric_summary"].get("candidate_mean"), (int, float))]
    if len(comparable) == TARGET_OUTCOMES and len({row["primary_metric_summary"].get("metric_id") for row in comparable}) == 1 and len({row["primary_metric_summary"].get("objective") for row in comparable}) == 1:
        objective = comparable[0]["primary_metric_summary"]["objective"]
        ordered = sorted(comparable, key=lambda row: ((-row["primary_metric_summary"]["candidate_mean"]) if objective == "maximize" else row["primary_metric_summary"]["candidate_mean"], row["attempt_id"]))
        selection = {"status": "selected", "best_attempt_id": ordered[0]["attempt_id"], "reason": "pre-registered primary metric with deterministic tie-break"}
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
    required = {"DirectionSelected": {"direction"}, "VariantPlanned": {"variant", "feedback_from_attempt_ids"}, "AttemptReserved": {"attempt"}, "AttemptTransitioned": {"attempt_id", "new_state", "phase", "phase_state"}, "AttemptImplementationRevised": {"attempt_id", "implementation_hash", "attempt_input_hash", "protocol_hash", "sample_manifest_hash", "runtime_config_hash", "evaluator_hash"}, "AttemptAbandoned": {"attempt_id", "reason"}, "AttemptDispositioned": {"attempt_id", "failure_class", "target_state", "artifact_hashes", "route_outcome"}, "AttemptFinalized": {"trial_result", "route_outcome", "aggregate"}, "AuditMarker": {"index"}}[event_type]
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


def _apply_transition(attempt: dict[str, Any], new_state: str, phase: str | None, phase_state: str | None, timestamp: str) -> None:
    current = attempt["state"]
    if current in TERMINAL_ATTEMPT_STATES or new_state not in TRANSITIONS.get(current, set()):
        raise IntegrityError(f"illegal attempt transition {current} -> {new_state}")
    if (phase is None) != (phase_state is None):
        raise IntegrityError("phase and phase_state must be provided together")
    if phase is not None:
        if phase not in {"proxy", "full"} or phase_state not in {"PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED"}:
            raise IntegrityError("invalid phase transition payload")
        attempt["phases"][phase] = phase_state
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
