"""Strict S3 gate over event-derived authoritative state."""

from __future__ import annotations

from auto_research.config import bootstrap_profile_enabled
from auto_research.research_state import IntegrityError, ResearchEventLedger
from auto_research.s3_validation import S3ValidationError, validate_committed_s3

from .base import StageGateValidator


class S3GateValidator(StageGateValidator):
    stage_key = "S3_experiment"
    validator_name = "s3_experiment_gate_v3"

    def validate(self):
        errors: list[str] = []
        ledger = ResearchEventLedger(self.project_root)
        try:
            state = ledger.rebuild()
        except (ValueError, IntegrityError) as exc:
            errors.append(str(exc))
            state = {}
        route = state.get("last_route_outcome") if isinstance(state, dict) else None
        attempt_id = ((route or {}).get("source") or {}).get("attempt_id") if isinstance(route, dict) else None
        attempt = ((state.get("attempts") or {}).get(attempt_id)) if attempt_id else None
        trial = (state.get("trial_results") or {}).get(attempt_id) if isinstance(attempt, dict) else None
        if not isinstance(route, dict) or not isinstance(attempt, dict):
            errors.append("S3 requires one reducer-committed RouteOutcome bound to an attempt")
        else:
            direction_state = (state.get("directions") or {}).get(attempt.get("direction_semantic_hash"))
            direction = direction_state.get("spec") if isinstance(direction_state, dict) else None
            variant = (state.get("variants") or {}).get(attempt.get("variant_spec_hash"))
            trial_spec = attempt.get("frozen_trial_spec")
            proxy_payload = None
            committed_proxy = attempt.get("committed_proxy_outcome")
            if isinstance(committed_proxy, dict):
                event_id = committed_proxy.get("event_id")
                proxy_event = next(
                    (event for event in ledger.events() if event.get("event_id") == event_id),
                    None,
                )
                if isinstance(proxy_event, dict) and proxy_event.get("event_type") == "ProxyEvidenceCommitted":
                    proxy_payload = proxy_event.get("payload")
            if not isinstance(direction, dict) or not isinstance(variant, dict) or not isinstance(trial_spec, dict):
                errors.append("S3 authoritative DirectionSpec, VariantSpec, or frozen TrialSpec is missing")
            try:
                if not errors:
                    validate_committed_s3(
                        project_root=self.project_root,
                        direction=direction,
                        variant=variant,
                        state=state,
                        attempt=attempt,
                        route_outcome=route,
                        trial_spec=trial_spec,
                        trial_result=trial,
                        proxy_event_payload=proxy_payload,
                    )
            except S3ValidationError as exc:
                errors.append(str(exc))

        if errors:
            self.retry_check("s3_authoritative_transaction", "S3 authoritative state validation failed", details={"errors": errors[:20]})
        else:
            self.pass_check("s3_authoritative_transaction", artifact="meta/research_events.sqlite3")

        if bootstrap_profile_enabled(self.config) and isinstance(attempt, dict):
            budget = dict((((state.get("directions") or {}).get(attempt.get("direction_semantic_hash")) or {}).get("budget") or {}))
            manifest = trial.get("evidence_manifest") if isinstance(trial, dict) else None
            evidence_kinds = {
                entry.get("kind")
                for entry in (manifest or {}).get("entries") or []
                if isinstance(entry, dict)
            }
            if (
                attempt.get("profile") == "bootstrap"
                and attempt.get("method_evaluable") is True
                and attempt.get("state") == "METHOD_COMPLETED"
                and attempt.get("attempt_kind") == "bootstrap_proxy"
                and attempt.get("consumes_direction_budget") is False
                and attempt.get("reserved_slot") is False
                and isinstance(trial, dict)
                and trial.get("attempt_id") == attempt.get("attempt_id")
                and trial.get("method_evaluable") is True
                and trial.get("completeness") == "proxy"
                and "bootstrap_completion" in evidence_kinds
                and route.get("next_action") == "FINISH_RUN"
                and budget == {"target": 5, "reserved": 0, "consumed": 0}
            ):
                self.pass_check("bootstrap_proxy_complete", artifact="meta/research_events.sqlite3")
            elif route.get("next_action") in {"PAUSE_RESOURCE", "REPAIR_IMPLEMENTATION", "BLOCK_INTEGRITY"}:
                self.pass_check("bootstrap_failure_route")
            else:
                self.retry_check("bootstrap_proxy_complete", "bootstrap may finish only after one verified evaluable cheap proxy")
        return self.finalize()
