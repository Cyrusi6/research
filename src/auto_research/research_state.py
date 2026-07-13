"""Immutable event ledger and deterministic S1-S3 state reducer."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from .domain_contracts import (
    ATTEMPT_SCHEMA_VERSION,
    ROUTE_OUTCOME_SCHEMA_VERSION,
    TRIAL_RESULT_SCHEMA_VERSION,
    canonical_hash,
    validate_contract,
)
from .utils import ensure_dir, now_utc, write_json


EVENT_SCHEMA_VERSION = "auto_research_event_v1"
STATE_SCHEMA_VERSION = "auto_research_state_v1"
TERMINAL_ATTEMPT_STATES = {"METHOD_COMPLETED", "METHOD_FAILED", "INTEGRITY_BLOCKED"}


class ResearchEventLedger:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.events_dir = ensure_dir(project_root / "meta" / "research_events")
        self.snapshot_path = project_root / "meta" / "research_state.json"
        self.attempts_dir = ensure_dir(project_root / "meta" / "attempts")

    def rebuild(self) -> dict[str, Any]:
        state = initial_state(self.project_root.name)
        for event in self.events():
            state = reduce_event(state, event)
        self._write_snapshot(state)
        return state

    def state(self) -> dict[str, Any]:
        return self.rebuild()

    def events(self) -> list[dict[str, Any]]:
        records = []
        for path in sorted(self.events_dir.glob("*.json")):
            records.append(json.loads(path.read_text(encoding="utf-8")))
        return records

    def append(self, event_type: str, payload: dict[str, Any], *, event_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        event_id = event_id or str(uuid.uuid4())
        existing = self._event_by_id(event_id)
        if existing is not None:
            return existing, self.state()
        sequence = self._next_sequence()
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": event_id,
            "sequence": sequence,
            "event_type": event_type,
            "created_at": now_utc(),
            "payload": deepcopy(payload),
        }
        next_state = reduce_event(self.state(), event)
        target = self.events_dir / f"{sequence:012d}-{event_id}.json"
        self._atomic_json(target, event)
        self._write_snapshot(next_state)
        self._write_attempt_views(next_state)
        return event, next_state

    def select_direction(self, direction: dict[str, Any], *, event_id: str | None = None) -> dict[str, Any]:
        _, state = self.append("DirectionSelected", {"direction": direction}, event_id=event_id)
        return state

    def plan_variant(self, variant: dict[str, Any], *, feedback_from_attempt_ids: list[str] | None = None, event_id: str | None = None) -> dict[str, Any]:
        payload = {"variant": variant, "feedback_from_attempt_ids": feedback_from_attempt_ids or []}
        _, state = self.append("VariantPlanned", payload, event_id=event_id)
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
        event_id: str | None = None,
    ) -> dict[str, Any]:
        state = self.state()
        for attempt in state["attempts"].values():
            if (
                attempt["direction_hash"] == direction["direction_hash"]
                and attempt["variant_spec_hash"] == variant["variant_spec_hash"]
                and not attempt["method_evaluable"]
                and attempt["state"] != "INTEGRITY_BLOCKED"
            ):
                if attempt["implementation_hash"] != implementation_hash or attempt["attempt_input_hash"] != attempt_input_hash:
                    return self.transition_attempt(
                        attempt["attempt_id"],
                        "IMPLEMENTATION_REPAIR",
                        changes={"implementation_hash": implementation_hash, "attempt_input_hash": attempt_input_hash},
                        event_id=f"{event_id}:implementation-update" if event_id else None,
                    )
                return attempt
        budget = state["directions"].get(direction["direction_hash"], {}).get("budget", {})
        consumed = int(budget.get("consumed", 0))
        reserved = int(budget.get("reserved", 0))
        counts = profile == "standard" and attempt_kind != "bootstrap_proxy"
        if counts and consumed + reserved >= 5:
            raise RuntimeError("direction outcome budget exhausted; refusing sixth variant attempt")
        now = now_utc()
        attempt = {
            "schema_version": ATTEMPT_SCHEMA_VERSION,
            "attempt_id": str(uuid.uuid4()),
            "profile": profile,
            "direction_id": direction["direction_id"],
            "direction_hash": direction["direction_hash"],
            "variant_id": variant["variant_id"],
            "variant_spec_hash": variant["variant_spec_hash"],
            "implementation_hash": implementation_hash,
            "attempt_input_hash": attempt_input_hash,
            "attempt_kind": attempt_kind,
            "state": "PLANNED",
            "method_evaluable": False,
            "consumes_direction_budget": False,
            "reserved_slot": counts,
            "phases": {"proxy": "PENDING", "full": "PENDING"},
            "terminal_outcome": None,
            "failure_class": None,
            "artifact_hashes": {},
            "created_at": now,
            "updated_at": now,
        }
        validate_contract(attempt, "attempt_record_v1.schema.json")
        _, next_state = self.append("AttemptReserved", {"attempt": attempt}, event_id=event_id)
        return next_state["attempts"][attempt["attempt_id"]]

    def transition_attempt(self, attempt_id: str, state_name: str, *, changes: dict[str, Any] | None = None, event_id: str | None = None) -> dict[str, Any]:
        _, state = self.append(
            "AttemptStateChanged",
            {"attempt_id": attempt_id, "state": state_name, "changes": changes or {}},
            event_id=event_id,
        )
        return state["attempts"][attempt_id]

    def complete_attempt(self, trial_result: dict[str, Any], *, event_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        validate_contract(trial_result, "trial_result_v1.schema.json")
        event_id = event_id or f"attempt-completed:{trial_result['attempt_id']}:{canonical_hash(trial_result)}"
        _, state = self.append("AttemptCompleted", {"trial_result": trial_result}, event_id=event_id)
        route = route_after_attempt(state, trial_result["attempt_id"])
        route_event_id = f"route:{trial_result['attempt_id']}:{trial_result['outcome_classification']}"
        _, state = self.append("RouteCommitted", {"route_outcome": route}, event_id=route_event_id)
        write_json(self.project_root / "meta" / "route_outcome.json", route)
        return state["attempts"][trial_result["attempt_id"]], route

    def force_new_direction(self, *, reason_codes: list[str], event_id: str | None = None) -> dict[str, Any]:
        state = self.state()
        route = build_route_outcome(state, "START_NEW_DIRECTION", reason_codes=reason_codes, source_event_id=event_id)
        _, next_state = self.append("RouteCommitted", {"route_outcome": route}, event_id=event_id)
        write_json(self.project_root / "meta" / "route_outcome.json", route)
        return next_state

    def _next_sequence(self) -> int:
        paths = sorted(self.events_dir.glob("*.json"))
        return int(paths[-1].name.split("-", 1)[0]) + 1 if paths else 1

    def _event_by_id(self, event_id: str) -> dict[str, Any] | None:
        for path in self.events_dir.glob(f"*-{event_id}.json"):
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _write_snapshot(self, state: dict[str, Any]) -> None:
        snapshot = deepcopy(state)
        snapshot["updated_at"] = now_utc()
        self._atomic_json(self.snapshot_path, snapshot)

    def _write_attempt_views(self, state: dict[str, Any]) -> None:
        for attempt in state["attempts"].values():
            self._atomic_json(self.attempts_dir / f"{attempt['attempt_id']}.json", attempt)

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        ensure_dir(path.parent)
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def initial_state(project_id: str) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "project_id": project_id,
        "last_sequence": 0,
        "current_direction_hash": None,
        "current_variant_spec_hash": None,
        "directions": {},
        "variants": {},
        "attempts": {},
        "trial_results": {},
        "method_tried_history": [],
        "implementation_history": [],
        "last_route_outcome": None,
        "updated_at": now_utc(),
    }


def reduce_event(state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    next_state = deepcopy(state)
    sequence = int(event["sequence"])
    if sequence <= int(next_state.get("last_sequence", 0)):
        return next_state
    payload = event["payload"]
    event_type = event["event_type"]
    if event_type == "DirectionSelected":
        direction = deepcopy(payload["direction"])
        key = direction["direction_hash"]
        next_state["directions"].setdefault(
            key,
            {"spec": direction, "budget": {"target": 5, "reserved": 0, "consumed": 0}, "best_variant_id": None},
        )
        next_state["current_direction_hash"] = key
        next_state["current_variant_spec_hash"] = None
    elif event_type == "VariantPlanned":
        variant = deepcopy(payload["variant"])
        variant["feedback_from_attempt_ids"] = list(payload.get("feedback_from_attempt_ids") or [])
        next_state["variants"][variant["variant_spec_hash"]] = variant
        next_state["current_variant_spec_hash"] = variant["variant_spec_hash"]
    elif event_type == "AttemptReserved":
        attempt = deepcopy(payload["attempt"])
        next_state["attempts"][attempt["attempt_id"]] = attempt
        if attempt["reserved_slot"]:
            next_state["directions"][attempt["direction_hash"]]["budget"]["reserved"] += 1
    elif event_type == "AttemptStateChanged":
        attempt = next_state["attempts"][payload["attempt_id"]]
        previous_hash = attempt["implementation_hash"]
        attempt.update(deepcopy(payload.get("changes") or {}))
        attempt["state"] = payload["state"]
        attempt["updated_at"] = event["created_at"]
        if attempt["implementation_hash"] != previous_hash:
            next_state["implementation_history"].append(
                {
                    "attempt_id": attempt["attempt_id"],
                    "variant_spec_hash": attempt["variant_spec_hash"],
                    "previous_implementation_hash": previous_hash,
                    "implementation_hash": attempt["implementation_hash"],
                }
            )
    elif event_type == "AttemptCompleted":
        result = deepcopy(payload["trial_result"])
        attempt = next_state["attempts"][result["attempt_id"]]
        already_terminal = attempt["state"] in TERMINAL_ATTEMPT_STATES
        if not already_terminal:
            evaluable = bool(result["method_evaluable"])
            attempt["method_evaluable"] = evaluable
            attempt["consumes_direction_budget"] = bool(attempt["reserved_slot"] and evaluable)
            attempt["terminal_outcome"] = result["outcome_classification"]
            attempt["failure_class"] = result["failure_classification"]
            attempt["artifact_hashes"] = deepcopy(result["raw_artifacts"])
            attempt["state"] = "METHOD_COMPLETED" if evaluable else _non_evaluable_terminal_state(result)
            attempt["updated_at"] = event["created_at"]
            budget = next_state["directions"][attempt["direction_hash"]]["budget"]
            if attempt["reserved_slot"]:
                budget["reserved"] = max(0, budget["reserved"] - 1)
            if attempt["consumes_direction_budget"]:
                budget["consumed"] += 1
                next_state["method_tried_history"].append(
                    {
                        "attempt_id": attempt["attempt_id"],
                        "direction_hash": attempt["direction_hash"],
                        "variant_id": attempt["variant_id"],
                        "variant_spec_hash": attempt["variant_spec_hash"],
                        "outcome_classification": result["outcome_classification"],
                        "method_evaluable": True,
                    }
                )
            next_state["trial_results"][attempt["attempt_id"]] = result
    elif event_type == "RouteCommitted":
        next_state["last_route_outcome"] = deepcopy(payload["route_outcome"])
        action = payload["route_outcome"]["next_action"]
        if action == "FINISH_DIRECTION" and next_state.get("current_direction_hash"):
            accepted = [
                item for item in next_state["method_tried_history"]
                if item.get("direction_hash") == next_state["current_direction_hash"] and item.get("outcome_classification") == "accepted"
            ]
            if accepted:
                next_state["directions"][next_state["current_direction_hash"]]["best_variant_id"] = accepted[0]["variant_id"]
        if action == "START_NEW_DIRECTION":
            next_state["current_direction_hash"] = None
            next_state["current_variant_spec_hash"] = None
    next_state["last_sequence"] = sequence
    next_state["updated_at"] = event["created_at"]
    return next_state


def build_trial_result(
    *,
    attempt: dict[str, Any],
    protocol_hash: str,
    input_hash: str,
    completeness: str,
    observed_datasets: list[str],
    raw_artifacts: dict[str, str],
    proxy_observations: list[dict[str, Any]],
    full_observations: list[dict[str, Any]],
    ablation_observations: list[dict[str, Any]],
    method_evaluable: bool,
    outcome_classification: str,
    failure_classification: str | None,
) -> dict[str, Any]:
    result = {
        "schema_version": TRIAL_RESULT_SCHEMA_VERSION,
        "direction_id": attempt["direction_id"],
        "direction_hash": attempt["direction_hash"],
        "variant_id": attempt["variant_id"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "attempt_id": attempt["attempt_id"],
        "protocol_hash": protocol_hash,
        "attempt_input_hash": input_hash,
        "completeness": completeness,
        "observed_datasets": observed_datasets,
        "raw_artifacts": raw_artifacts,
        "proxy_observations": proxy_observations,
        "full_observations": full_observations,
        "ablation_observations": ablation_observations,
        "method_evaluable": method_evaluable,
        "outcome_classification": outcome_classification,
        "failure_classification": failure_classification,
    }
    validate_contract(result, "trial_result_v1.schema.json")
    return result


def route_after_attempt(state: dict[str, Any], attempt_id: str) -> dict[str, Any]:
    attempt = state["attempts"][attempt_id]
    if attempt["profile"] == "bootstrap":
        return build_route_outcome(state, "FINISH_RUN", reason_codes=["bootstrap_proxy_complete"], source_attempt_id=attempt_id)
    if not attempt["method_evaluable"]:
        if attempt["state"] == "RESOURCE_PAUSED":
            action = "PAUSE_RESOURCE"
        elif attempt["state"] == "INTEGRITY_BLOCKED":
            action = "BLOCK_INTEGRITY"
        else:
            action = "REPAIR_IMPLEMENTATION"
        return build_route_outcome(state, action, reason_codes=[attempt["failure_class"] or "non_evaluable_attempt"], source_attempt_id=attempt_id)
    direction = state["directions"][attempt["direction_hash"]]
    consumed = direction["budget"]["consumed"]
    if consumed < 5:
        return build_route_outcome(state, "PROPOSE_NEXT_VARIANT", reason_codes=["outcome_recorded", "direction_budget_remaining"], source_attempt_id=attempt_id)
    outcomes = [item for item in state["method_tried_history"] if item["direction_hash"] == attempt["direction_hash"]]
    accepted = [item for item in outcomes if item["outcome_classification"] == "accepted"]
    if accepted:
        direction["best_variant_id"] = accepted[0]["variant_id"]
        return build_route_outcome(state, "FINISH_DIRECTION", reason_codes=["five_outcomes_complete", "accepted_variant_exists"], source_attempt_id=attempt_id)
    return build_route_outcome(state, "START_NEW_DIRECTION", reason_codes=["five_outcomes_complete", "direction_falsified_or_no_acceptance"], source_attempt_id=attempt_id)


def build_route_outcome(
    state: dict[str, Any],
    next_action: str,
    *,
    reason_codes: list[str],
    source_attempt_id: str | None = None,
    source_event_id: str | None = None,
) -> dict[str, Any]:
    direction_hash = state.get("current_direction_hash")
    direction = state["directions"].get(direction_hash or "", {})
    variant = state["variants"].get(state.get("current_variant_spec_hash") or "", {})
    budget = deepcopy(direction.get("budget") or {"target": 5, "reserved": 0, "consumed": 0})
    outcome = {
        "schema_version": ROUTE_OUTCOME_SCHEMA_VERSION,
        "source": {"event_id": source_event_id, "attempt_id": source_attempt_id},
        "identity": {
            "direction_id": (direction.get("spec") or {}).get("direction_id"),
            "direction_hash": direction_hash,
            "variant_id": variant.get("variant_id"),
            "variant_spec_hash": variant.get("variant_spec_hash"),
            "attempt_id": source_attempt_id,
        },
        "next_action": next_action,
        "reason_codes": reason_codes,
        "budget_snapshot": budget,
        "artifact_hashes": {},
        "idempotency_key": canonical_hash(
            {"source_event_id": source_event_id, "source_attempt_id": source_attempt_id, "next_action": next_action, "budget": budget}
        ),
    }
    validate_contract(outcome, "route_outcome_v1.schema.json")
    return outcome


def _non_evaluable_terminal_state(result: dict[str, Any]) -> str:
    failure = result.get("failure_classification") or ""
    if failure in {"resource_paused", "oom_retry", "resource_unavailable"}:
        return "RESOURCE_PAUSED"
    if failure in {"integrity", "safety", "identity_mismatch"}:
        return "INTEGRITY_BLOCKED"
    return "METHOD_FAILED"
