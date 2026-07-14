"""Strict S3 gate over event-derived authoritative state."""

from __future__ import annotations

from auto_research.config import bootstrap_profile_enabled
from auto_research.domain_contracts import (
    contract_errors,
    validate_direction_identity,
    validate_trial_result,
    validate_variant_identity,
)
from auto_research.research_state import IntegrityError, ResearchEventLedger
from auto_research.utils import read_json

from .base import StageGateValidator, load_schema, validate_min_schema


class S3GateValidator(StageGateValidator):
    stage_key = "S3_experiment"
    validator_name = "s3_experiment_gate_v3"

    def validate(self):
        required = ["literature/direction.json", "plan/variant.json"]
        if not all(self.require_file(path, retry=True) for path in required):
            return self.finalize()
        direction = self.read_json_artifact("literature/direction.json")
        variant = self.read_json_artifact("plan/variant.json")
        if not isinstance(direction, dict) or not isinstance(variant, dict):
            self.retry_check("s3_authoritative_json", "DirectionSpec and VariantSpec must be JSON objects")
            return self.finalize()
        errors: list[str] = []
        try:
            validate_direction_identity(direction)
            validate_variant_identity(direction, variant)
            state = ResearchEventLedger(self.project_root).rebuild()
        except (ValueError, IntegrityError) as exc:
            errors.append(str(exc))
            state = {}
        route = state.get("last_route_outcome") if isinstance(state, dict) else None
        attempt_id = ((route or {}).get("source") or {}).get("attempt_id") if isinstance(route, dict) else None
        attempt = ((state.get("attempts") or {}).get(attempt_id)) if attempt_id else None
        if not isinstance(route, dict) or not isinstance(attempt, dict):
            errors.append("S3 requires one reducer-committed RouteOutcome bound to an attempt")
        else:
            errors.extend(contract_errors(route, "route_outcome_v2.schema.json"))
            for key in ["direction_id", "direction_semantic_hash", "direction_spec_hash"]:
                if route.get("identity", {}).get(key) != direction.get(key):
                    errors.append(f"RouteOutcome {key} mismatch")
            for key in ["variant_id", "variant_semantic_hash", "variant_spec_hash"]:
                if route.get("identity", {}).get(key) != variant.get(key):
                    errors.append(f"RouteOutcome {key} mismatch")
            budget = ((state.get("directions") or {}).get(attempt.get("direction_semantic_hash")) or {}).get("budget") or {}
            if budget.get("consumed", 0) < 0 or budget.get("reserved", 0) < 0 or budget.get("consumed", 0) + budget.get("reserved", 0) > 5:
                errors.append("direction budget invariant violated")

        trial = (state.get("trial_results") or {}).get(attempt_id) if isinstance(attempt, dict) else None
        if attempt and attempt.get("method_evaluable"):
            if not isinstance(trial, dict):
                errors.append("method-evaluable attempt is missing canonical TrialResult")
            else:
                try:
                    validate_trial_result(trial)
                except ValueError as exc:
                    errors.append(str(exc))
                errors.extend(contract_errors(trial, "trial_result_v2.schema.json"))
                for key in [
                    "direction_id", "direction_semantic_hash", "direction_spec_hash",
                    "variant_id", "variant_semantic_hash", "variant_spec_hash", "attempt_id",
                    "attempt_input_hash", "protocol_hash",
                ]:
                    if trial.get(key) != attempt.get(key):
                        errors.append(f"TrialResult {key} mismatch")
        elif trial is not None:
            errors.append("non-evaluable attempt cannot own a canonical TrialResult")

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
        self._validate_proxy_contracts_if_present()
        return self.finalize()

    def _validate_proxy_contracts_if_present(self):
        artifacts = {
            "baseline_fingerprint": ("experiment/results/c2c_proxy_baseline_fingerprint.json", "c2c_proxy_baseline_fingerprint.schema.json"),
            "cache_report": ("experiment/results/c2c_proxy_cache_report.json", "c2c_proxy_cache_report.schema.json"),
            "effective_policy": ("experiment/results/c2c_effective_proxy_policy.json", "c2c_effective_proxy_policy.schema.json"),
            "decision_report": ("experiment/results/c2c_proxy_decision_report.json", "c2c_proxy_decision_report.schema.json"),
            "calibration_policy": ("experiment/results/c2c_proxy_calibration_policy.json", "c2c_proxy_calibration_policy.schema.json"),
        }
        if not any((self.project_root / path).exists() for path, _ in artifacts.values()):
            return
        for name, (path, schema_name) in artifacts.items():
            if not (self.project_root / path).exists():
                continue
            schema_errors = validate_min_schema(self.read_json_artifact(path), load_schema(schema_name))
            if schema_errors:
                self.retry_check(f"{name}_schema", f"{path} failed schema validation", details={"errors": schema_errors[:20]})
            else:
                self.pass_check(f"{name}_schema", artifact=path)
