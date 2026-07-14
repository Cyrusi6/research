"""Strict S3 gate over event-derived authoritative state."""

from __future__ import annotations

from auto_research.config import bootstrap_profile_enabled
from auto_research.research_state import IntegrityError, ResearchEventLedger
from auto_research.s3_validation import S3ValidationError, validate_committed_s3
from auto_research.utils import read_json

from .base import StageGateValidator


class S3GateValidator(StageGateValidator):
    stage_key = "S3_experiment"
    validator_name = "s3_experiment_gate_v3"

    def validate(self):
        required = ["literature/direction.json", "plan/variant.json"]
        if not all(self.require_file(path, retry=True) for path in required):
            return self.finalize()
        if not self.require_file("plan/trial_spec.json", retry=True):
            self.retry_check("s3_authoritative_transaction", "S3 pre/post-commit audit requires canonical plan/trial_spec.json")
            return self.finalize()
        direction = self.read_json_artifact("literature/direction.json")
        variant = self.read_json_artifact("plan/variant.json")
        trial_spec = self.read_json_artifact("plan/trial_spec.json")
        if not isinstance(direction, dict) or not isinstance(variant, dict) or not isinstance(trial_spec, dict):
            self.retry_check("s3_authoritative_json", "DirectionSpec, VariantSpec, and TrialSpec must be JSON objects")
            return self.finalize()
        errors: list[str] = []
        try:
            state = ResearchEventLedger(self.project_root).rebuild()
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
            try:
                validate_committed_s3(
                    project_root=self.project_root,
                    direction=direction,
                    variant=variant,
                    state=state,
                    attempt=attempt,
                    route_outcome=route,
                    trial_spec=trial_spec,
                    trial_result=trial,
                )
            except S3ValidationError as exc:
                errors.append(str(exc))

        if errors:
            self.retry_check("s3_authoritative_transaction", "S3 authoritative state validation failed", details={"errors": errors[:20]})
        else:
            self.pass_check("s3_authoritative_transaction", artifact="meta/research_events.sqlite3")

        if bootstrap_profile_enabled(self.config) and isinstance(attempt, dict):
            completion = read_json(self.project_root / "experiment" / "results" / "bootstrap_proxy_completion.json", default={}) or {}
            if (
                attempt.get("method_evaluable") is True
                and attempt.get("attempt_kind") == "bootstrap_proxy"
                and attempt.get("consumes_direction_budget") is False
                and route.get("next_action") == "FINISH_RUN"
                and completion.get("bootstrap_proxy_complete") is True
            ):
                self.pass_check("bootstrap_proxy_complete", artifact="experiment/results/bootstrap_proxy_completion.json")
            elif route.get("next_action") in {"PAUSE_RESOURCE", "REPAIR_IMPLEMENTATION", "BLOCK_INTEGRITY"}:
                self.pass_check("bootstrap_failure_route")
            else:
                self.retry_check("bootstrap_proxy_complete", "bootstrap may finish only after one verified evaluable cheap proxy")
        return self.finalize()
